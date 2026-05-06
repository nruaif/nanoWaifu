import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import math
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torchvision.io as tv_io

# ==========================================================
# 1. Positional Encoding & Standard Blocks
# ==========================================================
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0):
    y_pos = torch.arange(height, dtype=torch.float32)
    x_pos = torch.arange(width, dtype=torch.float32)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis[None, :, None, :]
    elif freqs_cis.ndim == 3:
        freqs_cis = freqs_cis[:, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        # Q is 2x size, K and V are 1x size
        self.qkv = nn.Linear(dim, dim * 4, bias=False)
        self.lambda_proj = nn.Linear(dim, heads, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, pos, attn_mask=None):
        B, N, C = x.shape

        # 1. Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split([C * 2, C, C], dim=-1)

        # q has 2 * heads
        q = q.reshape(B, N, 2 * self.heads, self.head_dim)
        k = k.reshape(B, N, self.heads, self.head_dim)
        v = v.reshape(B, N, self.heads, self.head_dim)

        # 2. Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)

        # Transpose for attention: (Batch, Heads, SeqLen, HeadDim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. Handle GQA broadcasting for PyTorch SDPA
        # (If using the official `flash_attn_func` from Dao-AILab, you don't need this repeat,
        # but PyTorch's native F.scaled_dot_product_attention requires broadcastable shapes).
        # We interleave K and V so head 0 and 1 of Q share head 0 of K/V.
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)

        # 4. A SINGLE ATTENTION CALL
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # 5. Split post-attention (Even heads = attn1, Odd heads = attn2)
        attn1 = attn[:, 0::2]
        attn2 = attn[:, 1::2]

        # 6. Calculate Differential Scalar λ
        lambda_ = torch.sigmoid(self.lambda_proj(x))  # (B, N, heads)
        lambda_ = lambda_.transpose(1, 2).unsqueeze(-1)  # (B, heads, N, 1)

        # 7. Differential Subtraction
        diff_out = attn1 - lambda_ * attn2

        # 8. Final Output Projection
        diff_out = diff_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(diff_out)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(2 * (dim * 4) / 3)
        self.mlp = SwiGLU(dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, c, pos, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)), pos, attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)))
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ==========================================================
# 2. DeCo Pixel Decoder (Predicts x1)
# ==========================================================
class DeCoDecoderBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, 3 * dim, bias=True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True), nn.SiLU(), nn.Linear(4 * dim, dim, bias=True)
        )

    def forward(self, h, c_aligned):
        alpha, beta, gamma = self.adaLN_modulation(F.silu(c_aligned)).chunk(3, dim=-1)
        h_norm = self.norm(h)
        h_modulated = h_norm * gamma + beta
        h_out = self.mlp(h_modulated)
        return h + alpha * h_out


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs=8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs ** 2, hidden_size_input, bias=True))
        self.precompute_pos = {}

    def fetch_pos(self, height, width, device, dtype):
        if (height, width) not in self.precompute_pos:
            pos_y = torch.linspace(0, 1, height, device=device, dtype=dtype)
            pos_x = torch.linspace(0, 1, width, device=device, dtype=dtype)
            pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
            pos_x, pos_y = pos_x.reshape(-1, 1, 1), pos_y.reshape(-1, 1, 1)
            freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
            freqs_x, freqs_y = freqs[None, :, None], freqs[None, None, :]
            coeffs = (1 + freqs_x * freqs_y) ** -1
            dct_x, dct_y = torch.cos(pos_x * freqs_x * torch.pi), torch.cos(pos_y * freqs_y * torch.pi)
            dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
            self.precompute_pos[(height, width)] = dct
        return self.precompute_pos[(height, width)].to(device)

    def forward(self, inputs, height, width):
        B, L, C = inputs.shape
        dct = self.fetch_pos(height, width, inputs.device, inputs.dtype).expand(B, -1, -1)
        return self.embedder(torch.cat([inputs, dct], dim=-1))


class DeCoPixelDecoder(nn.Module):
    def __init__(self, in_channels, depth=3, cond_dim=1024):
        super().__init__()
        self.dim = in_channels
        self.w_in = NerfEmbedder(in_channels, self.dim)
        self.blocks = nn.ModuleList([DeCoDecoderBlock(self.dim, cond_dim) for _ in range(depth)])
        self.out_proj = nn.Linear(self.dim, in_channels)

    def forward(self, x_raw, c_low_freq, height, width, t_emb=None):
        c_aligned = c_low_freq
        if t_emb is not None:
            c_aligned = c_aligned + t_emb.unsqueeze(1)
        h = self.w_in(x_raw, height, width)
        for block in self.blocks:
            h = block(h, c_aligned)
        return self.out_proj(h)


# ==========================================================
# 3. MAE TokenformerDiT Architecture
# ==========================================================
class MAE_TokenformerDiT(nn.Module):
    def __init__(
            self, in_channels=3, base_channels=1024, num_blocks=24, heads=16,
            patch_size=16, num_classes=1000, use_deco_decoder=False,
            cond_drop_prob=0.1, **kwargs
    ):
        super().__init__()

        # Architecture parameters (MAE Asymmetric Scaling)
        self.encoder_dim = kwargs.get('dim', base_channels)
        self.encoder_depth = kwargs.get('depth', num_blocks)

        self.decoder_dim = self.encoder_dim // 2
        self.decoder_depth = max(1, self.encoder_depth // 2)
        dec_heads = max(1, heads // 2)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patchified_channels = in_channels * patch_size * patch_size
        self.use_deco_decoder = use_deco_decoder
        self.cond_drop_prob = cond_drop_prob

        # Embeddings
        self.input_proj = nn.Conv2d(self.patchified_channels, self.encoder_dim, 1)
        self.t_embedder = TimestepEmbedder(self.encoder_dim)

        if num_classes > 0:
            self.num_classes = num_classes
            self.y_embedder = nn.Embedding(num_classes, self.encoder_dim)
            self.y_proj = nn.Linear(num_classes, self.encoder_dim)
            self.null_y_token = nn.Parameter(torch.normal(0, 0.02, size=(1, 1, self.encoder_dim)))
        else:
            self.y_embedder = self.y_proj = self.null_y_token = None

        # --- ENCODER --- (Only processes visible tokens)
        self.encoder_blocks = nn.ModuleList([DiTBlock(self.encoder_dim, heads) for _ in range(self.encoder_depth)])
        self.encoder_norm = RMSNorm(self.encoder_dim)

        # --- TRANSITION ---
        self.enc_to_dec = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_enc_to_dec = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))

        # --- DECODER --- (Processes visible + mask tokens)
        self.decoder_blocks = nn.ModuleList([DiTBlock(self.decoder_dim, dec_heads) for _ in range(self.decoder_depth)])
        self.decoder_norm = RMSNorm(self.decoder_dim)

        self.conf_proj = nn.Linear(self.decoder_dim, 1) if self.use_deco_decoder else nn.Conv2d(self.decoder_dim, 1, 1)

        if self.use_deco_decoder:
            self.pixel_decoder = DeCoPixelDecoder(in_channels=self.patchified_channels, depth=3,
                                                  cond_dim=self.decoder_dim)
        else:
            self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(self.decoder_dim, 2 * self.decoder_dim, bias=True))
            self.out_proj = nn.Conv2d(self.decoder_dim, self.patchified_channels, 1)

        self.precompute_pos = {}

    def fetch_pos(self, height, width, device):
        if (height, width) not in self.precompute_pos:
            # RoPE head dim is consistent across encoder and decoder
            # (encoder_dim // heads == decoder_dim // dec_heads)
            pos = precompute_freqs_cis_2d(self.encoder_dim // self.encoder_blocks[0].attn.heads, height, width).to(
                device)
            self.precompute_pos[(height, width)] = pos
        return self.precompute_pos[(height, width)].to(device)

    def forward_dit(self, x, t, y=None, mask=None, drop_mask=None):
        B, C, H, W = x.shape
        x = self.input_proj(x).flatten(2).transpose(1, 2)
        pos = self.fetch_pos(H, W, x.device)

        t_emb = self.t_embedder(t)
        c_enc = t_emb

        if mask is None:
            mask = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device)

        # ---------------------------------------------------------
        # MAE STEP 1: Shuffle & Extract Visible Tokens
        # ---------------------------------------------------------
        ids_shuffle = torch.argsort(mask.int(), dim=1)  # False(0) comes first, True(1) masked last
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Gather standard features based on mask sort order
        x_shuffled = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).expand(-1, -1, self.encoder_dim))
        pos_b = pos.unsqueeze(0).expand(B, -1, -1)
        pos_shuffled = torch.gather(pos_b, dim=1, index=ids_shuffle.unsqueeze(-1).expand(-1, -1, pos.shape[-1]))
        mask_shuffled = torch.gather(mask, dim=1, index=ids_shuffle)

        # To handle variable masked sequences in a batch, we pad to the max visible length
        L_vis_max = (~mask).sum(dim=1).max().item()
        L_vis_max = max(1, L_vis_max)  # Fallback if sequence is entirely masked

        x_vis = x_shuffled[:, :L_vis_max, :]
        pos_vis = pos_shuffled[:, :L_vis_max, :]
        enc_pad_mask = mask_shuffled[:, :L_vis_max]  # True means it is padding (was actually a mask token)

        # Class Conditioning Handle
        num_y_tokens = 0
        if y is not None and self.y_embedder is not None:
            y_seq = self.y_embedder(y) if y.dtype in [torch.int, torch.int32, torch.long, torch.int64] else self.y_proj(
                y.float()).unsqueeze(1)
            if y_seq.ndim == 2: y_seq = y_seq.unsqueeze(1)
            num_y_tokens = y_seq.shape[1]

            if drop_mask is None and self.training and self.cond_drop_prob > 0:
                drop_mask = torch.rand(B, device=x.device) < self.cond_drop_prob
            if drop_mask is not None and drop_mask.any():
                num_dropped = drop_mask.sum().item()
                y_seq[drop_mask] = self.null_y_token.expand(num_dropped, num_y_tokens, -1)

            # Prepend class tokens and position padding to visible seq
            x_vis = torch.cat([y_seq, x_vis], dim=1)
            pos_pad = torch.ones((B, num_y_tokens, pos_vis.shape[-1]), dtype=pos_vis.dtype, device=pos_vis.device)
            pos_vis = torch.cat([pos_pad, pos_vis], dim=1)

            # y tokens are always valid, so padding mask is False
            y_pad_mask = torch.zeros((B, num_y_tokens), dtype=torch.bool, device=x.device)
            enc_pad_mask = torch.cat([y_pad_mask, enc_pad_mask], dim=1)

        # Create precise attention mask for F.scaled_dot_product_attention
        # (SDPA uses True to Allow, False to Block)
        if enc_pad_mask.any():
            attn_mask = (~enc_pad_mask).unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, Seq)
        else:
            attn_mask = None

        # ---------------------------------------------------------
        # MAE STEP 2: Encode Visible
        # ---------------------------------------------------------
        x_enc = x_vis
        for blk in self.encoder_blocks:
            x_enc = blk(x_enc, c_enc, pos_vis, attn_mask=attn_mask)
        x_enc = self.encoder_norm(x_enc)

        # Break off y tokens so we can properly unshuffle the image grid
        if num_y_tokens > 0:
            y_enc = x_enc[:, :num_y_tokens, :]
            x_enc_patches = x_enc[:, num_y_tokens:, :]
        else:
            x_enc_patches = x_enc

        # ---------------------------------------------------------
        # MAE STEP 3: Transition & Reconstruct Grid
        # ---------------------------------------------------------
        x_dec_vis = self.enc_to_dec(x_enc_patches)
        c_dec = self.c_enc_to_dec(c_enc)

        B, L = mask.shape
        x_dec_full_shuffled = self.mask_token.expand(B, L, -1).clone()

        # Scatter the encoded visible patches into the correct (still sorted) slots
        valid_vis_mask = (~mask_shuffled[:, :L_vis_max]).unsqueeze(-1)
        x_dec_full_shuffled[:, :L_vis_max, :] = torch.where(
            valid_vis_mask,
            x_dec_vis,
            x_dec_full_shuffled[:, :L_vis_max, :]
        )

        # Unshuffle tokens to their true original layout
        x_dec = torch.gather(x_dec_full_shuffled, dim=1,
                             index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim))

        # Re-attach y tokens for the decoder and reset full positional embedding
        if num_y_tokens > 0:
            y_dec = self.enc_to_dec(y_enc)
            x_dec = torch.cat([y_dec, x_dec], dim=1)
            pos_pad_dec = torch.ones((B, num_y_tokens, pos.shape[-1]), dtype=pos.dtype, device=pos.device)
            pos_dec = torch.cat([pos_pad_dec, pos_b], dim=1)
        else:
            pos_dec = pos_b

        # ---------------------------------------------------------
        # MAE STEP 4: Decode Full Output
        # ---------------------------------------------------------
        for blk in self.decoder_blocks:
            x_dec = blk(x_dec, c_dec, pos_dec, attn_mask=None)  # No padding needed here

        c_low_freq = self.decoder_norm(x_dec)

        if num_y_tokens > 0:
            c_low_freq = c_low_freq[:, num_y_tokens:, :]

        if self.use_deco_decoder:
            conf = self.conf_proj(c_low_freq).squeeze(-1)
        else:
            shift, scale = self.final_adaLN(c_dec).chunk(2, dim=-1)
            x_mod = modulate(c_low_freq, shift.unsqueeze(1), scale.unsqueeze(1))
            x_spatial = x_mod.transpose(1, 2).reshape(B, -1, H, W)
            conf = self.conf_proj(x_spatial).squeeze(1)

        return c_low_freq, conf, c_dec

    def forward(self, x, t, y=None, mask=None):
        B, C, H, W = x.shape
        x_raw = x.flatten(2).transpose(1, 2)

        c_low_freq, conf, c_dec = self.forward_dit(x, t, y, mask)

        if self.use_deco_decoder:
            x1_pred = self.pixel_decoder(x_raw, c_low_freq, H, W, t_emb=c_dec)
            return x1_pred.transpose(1, 2).reshape(B, -1, H, W), conf.reshape(B, H, W)
        else:
            shift, scale = self.final_adaLN(c_dec).chunk(2, dim=-1)
            x_mod = modulate(c_low_freq, shift.unsqueeze(1), scale.unsqueeze(1)).transpose(1, 2).reshape(B, -1, H, W)
            return self.out_proj(x_mod), conf

    @torch.no_grad()
    def sample(self, B, H, W, device, maskgit_steps=10, deco_steps=50, y=None, cfg_scale=4.0):
        assert self.use_deco_decoder, "Cascaded sampling requires DeCo Decoder enabled."

        do_cfg = cfg_scale > 1.0 and y is not None

        canvas_flat = torch.zeros(B, H * W, self.patchified_channels, device=device)
        mask = torch.ones(B, H * W, dtype=torch.bool, device=device)
        seq_len = H * W
        t_dit = torch.zeros(B, device=device)

        for i in range(maskgit_steps):
            canvas_spatial = canvas_flat.transpose(1, 2).reshape(B, -1, H, W)

            if do_cfg:
                canvas_in = canvas_spatial.repeat(2, 1, 1, 1)
                t_in = t_dit.repeat(2)
                mask_in = mask.repeat(2, 1)
                y_in = y.repeat(2, 1) if y.ndim > 1 else y.repeat(2)

                drop_mask = torch.cat([torch.zeros(B, dtype=torch.bool, device=device),
                                       torch.ones(B, dtype=torch.bool, device=device)], dim=0)

                c_low_both, conf_both, _ = self.forward_dit(canvas_in, t_in, y=y_in, mask=mask_in, drop_mask=drop_mask)
                c_low_cond, c_low_uncond = c_low_both.chunk(2, dim=0)
                conf_cond, conf_uncond = conf_both.chunk(2, dim=0)

                conf = conf_uncond + cfg_scale * (conf_cond - conf_uncond)
            else:
                c_low_freq, conf, _ = self.forward_dit(canvas_spatial, t_dit, y=y, mask=mask)
                c_low_cond = c_low_freq

            ratio = math.cos(((i + 1) / maskgit_steps) * (math.pi / 2))
            n_masked = int(ratio * seq_len)

            gumbel_noise = -torch.log(-torch.log(torch.rand_like(conf) + 1e-9) + 1e-9)
            scores = conf + 0.1 * gumbel_noise

            scores = torch.where(mask, scores, torch.full_like(scores, -float('inf')))
            _, sorted_indices = torch.sort(scores, dim=-1, descending=True)

            new_mask = torch.zeros_like(mask)
            if n_masked > 0:
                new_mask.scatter_(1, sorted_indices[:, :n_masked], True)

            just_unmasked = mask & (~new_mask)

            if just_unmasked.any():
                noise = torch.randn_like(canvas_flat)
                just_unmasked_expanded = just_unmasked.unsqueeze(-1)
                active_x_raw = torch.where(just_unmasked_expanded, noise, canvas_flat)

                dt = 1.0 / deco_steps
                for step in range(deco_steps):
                    t_val = step * dt
                    t_deco = torch.full((B,), t_val, device=device)
                    # Project condition embedding to decoder logic
                    t_emb_deco = self.c_enc_to_dec(self.t_embedder(t_deco * 1000))

                    if do_cfg:
                        x_in = active_x_raw.repeat(2, 1, 1)
                        c_in = torch.cat([c_low_cond, c_low_uncond], dim=0)
                        t_emb_in = t_emb_deco.repeat(2, 1)

                        x1_pred_both = self.pixel_decoder(x_in, c_in, H, W, t_emb=t_emb_in)
                        x1_cond, x1_uncond = x1_pred_both.chunk(2, dim=0)

                        x1_pred = x1_uncond + cfg_scale * (x1_cond - x1_uncond)
                    else:
                        x1_pred = self.pixel_decoder(active_x_raw, c_low_cond, H, W, t_emb=t_emb_deco)

                    denom = max(1.0 - t_val, 1e-5)
                    v_pred = (x1_pred - active_x_raw) / denom

                    active_x_raw = torch.where(just_unmasked_expanded, active_x_raw + v_pred * dt, active_x_raw)

                canvas_flat = torch.where(just_unmasked_expanded, active_x_raw, canvas_flat)

            mask = new_mask
            if n_masked == 0:
                break

        return canvas_flat.transpose(1, 2).reshape(B, -1, H, W)


# ==========================================
# 4. Image Processing & Utilities
# ==========================================
def load_and_preprocess_image(image_path, device):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    scale = 256.0 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32

    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"Loaded image. Original size: {w}x{h}. Resized to: {new_w}x{new_h}")
    return img_tensor


def patchify(x, p=16):
    B, C, H, W = x.shape
    Hp, Wp = H // p, W // p
    x = x.reshape(B, C, Hp, p, Wp, p)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * p * p, Hp, Wp)
    return x


def unpatchify(x, p=16):
    B, C_pp, Hp, Wp = x.shape
    C = C_pp // (p * p)
    x = x.reshape(B, C, p, p, Hp, Wp)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C, Hp * p, Wp * p)
    return x


# ==========================================
# 5. Overfitting Script
# ==========================================
def train_overfit(image_path, iterations=200000, patch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x1_img = load_and_preprocess_image(image_path, device)
    x1 = patchify(x1_img, p=patch_size)

    B, C, Hp, Wp = x1.shape
    H_img, W_img = x1_img.shape[-2], x1_img.shape[-1]  # For upsampling the visualization

    model = MAE_TokenformerDiT(
        in_channels=3,
        base_channels=256,
        num_blocks=4,
        heads=8,
        patch_size=patch_size,
        use_deco_decoder=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    os.makedirs("overfit_outputs", exist_ok=True)
    save_image(x1_img * 0.5 + 0.5, "overfit_outputs/target_image.png")

    print(f"Starting MAE overfit loop... Target patch grid: {Hp}x{Wp}")
    model.train()

    # --- NEW: List to store our frames for the video ---
    logvar_frames = []

    for step in range(iterations):
        optimizer.zero_grad()

        t = torch.empty((B,), device=device).uniform_(0, 1.0)
        t_condition = t * 1000

        x0 = torch.randn_like(x1)

        t_expand = t.view(B, 1, 1, 1)
        x_t = t_expand * x1 + (1 - t_expand) * x0
        seq_len = Hp * Wp
        mask_ratio = torch.normal(mean=0.75, std=0.25, size=(B,), device=device)
        mask_ratio = torch.clamp(mask_ratio, min=0.0, max=1.0)
        n_masked = (mask_ratio * seq_len).long()
        mask = torch.zeros((B, seq_len), dtype=torch.bool, device=device)
        for b in range(B):
            if n_masked[b] > 0:
                perm = torch.randperm(seq_len, device=device)
                mask[b, perm[:n_masked[b]]] = True

        # Forward DiT on the CLEAN image (x1) to extract global context
        t_dit = torch.zeros((B,), device=device)
        c_low_freq, conf, c_dec_embed = model.forward_dit(x1, t_dit, y=None, mask=mask)

        # Forward Pixel Decoder on the NOISY image (x_t)
        t_emb = model.t_embedder(t_condition)
        t_emb_dec = model.c_enc_to_dec(t_emb)  # Project to MAE decoder dimension
        x_raw = x_t.flatten(2).transpose(1, 2)
        x1_pred_flat = model.pixel_decoder(x_raw, c_low_freq, Hp, Wp, t_emb=t_emb_dec)
        x1_pred = x1_pred_flat.transpose(1, 2).reshape(B, -1, Hp, Wp)

        # Extract confidence scores
        logvar_theta = conf.reshape(B, Hp, Wp)

        # Loss Calculation (in v-space)
        mse_loss_raw = F.mse_loss(x1_pred, x1, reduction='none')
        mse_loss_spatial = mse_loss_raw.mean(dim=1)

        mse_loss_sg = mse_loss_spatial.detach()
        nll_loss = 0.5 * (mse_loss_sg * torch.exp(-logvar_theta) + logvar_theta)

        loss_v = mse_loss_spatial.mean()
        loss_nll = nll_loss.mean()
        loss = loss_v + loss_nll * 0.01

        loss.backward()
        optimizer.step()

        # --- NEW: Capture logvar_theta for visualization every 20 steps ---
        if step % 20 == 0:
            with torch.no_grad():
                # Extract the 2D map for the first item in batch
                var_map = logvar_theta[0].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, Hp, Wp)

                # Upsample to match original image dimensions for a better-looking video
                var_map_up = F.interpolate(var_map, size=(H_img, W_img), mode='bilinear', align_corners=False).squeeze()

                # Normalize to [0, 1] range based on current min/max
                var_min, var_max = var_map_up.min(), var_map_up.max()
                if var_max > var_min:
                    var_norm = (var_map_up - var_min) / (var_max - var_min)
                else:
                    var_norm = var_map_up - var_min

                # Apply a colormap (viridis: purple=low variance/high confidence, yellow=high variance/low conf)
                cmap = plt.get_cmap('viridis')
                var_colored = cmap(var_norm.cpu().numpy())[..., :3]  # Drop Alpha channel

                # Convert to uint8 RGB tensor [H, W, 3] and append to frames
                var_colored_uint8 = (var_colored * 255).astype(np.uint8)
                logvar_frames.append(torch.from_numpy(var_colored_uint8))

        if step % 100 == 0:
            print(
                f"Step {step:04d} | Total Loss: {loss.item():.4f} | V-MSE: {loss_v.item():.4f} | NLL: {loss_nll.item():.4f}")

        if step > 0 and step % 500 == 0:
            model.eval()
            print(f"--> Generating MAE sample at step {step}...")
            sampled_patches = model.sample(
                B=1, H=Hp, W=Wp,
                device=device,
                maskgit_steps=25,
                deco_steps=50,
                y=None
            )

            sampled_img = unpatchify(sampled_patches, p=patch_size)
            sampled_img = torch.clamp(sampled_img * 0.5 + 0.5, 0, 1)
            save_image(sampled_img, f"overfit_outputs/sample_step_{step:04d}.png")
            model.train()

    print("Overfitting complete!")

    # --- NEW: Save the collected frames as a video ---
    if len(logvar_frames) > 0:
        print("Compiling logvar_theta evolution video...")
        # Stack frames into [T, H, W, C] format required by torchvision
        video_tensor = torch.stack(logvar_frames)
        video_path = "overfit_outputs/logvar_evolution.mp4"
        # Write out at 30 fps
        tv_io.write_video(video_path, video_tensor, fps=30)
        print(f"Saved visualization video to: {video_path}")


if __name__ == "__main__":
    image_file = "kurimi.jpg"

    if not os.path.exists(image_file):
        print("Creating dummy kurimi.jpg for testing...")
        dummy_img = torch.rand(3, 512, 768)
        save_image(dummy_img, image_file)

    train_overfit(image_file, iterations=3000)