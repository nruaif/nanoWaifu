import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
import random


# ─── Core Building Blocks ───────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), weight=self.weight, eps=self.eps)


class MeanPoolingEmbedder(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.embed = nn.EmbeddingBag(num_classes + 1, dim * 4, mode='mean')

    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor) -> torch.Tensor:
        x = self.embed(y_indices, y_offsets)
        return x.view(x.shape[0], 4, -1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hidden_dim = hidden_dim

    def forward(self, t):
        """t: N-D spatial map or scalar timesteps (e.g. (B,) or (B, N))."""
        t = t * 1000.0
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)

        # Changed to unsqueeze(-1) to smoothly support both 1D and N-D spatial timesteps
        emb = t.float().unsqueeze(-1) * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1).to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


# ─── Transformer Components ─────────────────────────────────────────────────


class DWConvMlp(nn.Module):
    """Gated MLP with depthwise conv2d for local spatial mixing."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        B, N, D = x.shape
        x = x.transpose(1, 2).view(B, D, H, W)
        x = self.dwconv(x)
        x = x.view(B, D, N).transpose(1, 2)
        x = x * self.act(gate)
        return self.fc2(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK-norm."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = DWConvMlp(dim, dim * 4)

    def forward(self, x, H, W, context=None):
        n_ctx = 0
        if context is not None:
            x_seq = torch.cat([context, x], dim=1)
            n_ctx = context.shape[1]
        else:
            x_seq = x

        x_seq = x_seq + self.self_attn(self.norm1(x_seq))

        if n_ctx > 0:
            context = x_seq[:, :n_ctx]
            x = x_seq[:, n_ctx:]
        else:
            x = x_seq

        x = x + self.mlp(self.norm2(x), H, W)
        return x, context


class REPAProjector(nn.Module):
    """Projects mid-block features (32x downsample) to DINOv3 ViT-L space (16x, dim=1024)."""

    def __init__(self, in_dim=768, out_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.dw_conv = nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim)
        self.norm = RMSNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x, Hm, Wm):
        """
        x: (B, Hm*Wm, in_dim) — mid-block tokens at 32x resolution
        Returns: (B, Hp*Wp, out_dim) — projected tokens at 16x resolution
        """
        x = self.proj(x)  # (B, Hm*Wm, out_dim*4)
        B = x.shape[0]
        x = x.transpose(1, 2).view(B, self.out_dim * 4, Hm, Wm)
        x = F.pixel_shuffle(x, 2)  # (B, out_dim, Hp, Wp)
        x = self.dw_conv(x)
        x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, out_dim)
        x = self.norm(x)
        return x


# ─── Small UNet (Pixel Predictor) ───────────────────────────────────────────


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj1 = nn.Conv2d(in_ch, out_ch, 1)
        self.dw1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.proj2 = nn.Conv2d(out_ch, out_ch, 1)
        self.dw2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.act(self.gn1(self.dw1(self.proj1(x))))
        h = self.act(self.gn2(self.dw2(self.proj2(h))))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """
    Convolutional UNet predicting clean image x0.
    Transformer features are concatenated at the H/16 bottleneck directly.
    """

    def __init__(self, in_channels, cond_dim, base_dim=64):
        super().__init__()
        out_channels = in_channels  # x0 only; logvar is predicted by transformer
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        # Encoder (4 stages → f16)
        self.enc0 = nn.Conv2d(in_channels, dims[0], 3, 1, 1)  # H
        self.down0 = nn.Conv2d(dims[0] * 4, dims[0], 1)  # H/2
        self.enc1 = UNetBlock(dims[0], dims[1])
        self.down1 = nn.Conv2d(dims[1] * 4, dims[1], 1)  # H/4
        self.enc2 = UNetBlock(dims[1], dims[2])
        self.down2 = nn.Conv2d(dims[2] * 4, dims[2], 1)  # H/8
        self.enc3 = UNetBlock(dims[2], dims[3])
        self.down3 = nn.Conv2d(dims[3] * 4, dims[3], 1)  # H/16

        # Bottleneck @ H/16: Combines UNet enc features with Transformer features
        self.mid = UNetBlock(dims[3] + cond_dim, dims[3])

        # Decoder (4 stages → back to H)
        self.up3 = nn.Conv2d(dims[3], dims[3] * 4, 1)
        self.dec3 = UNetBlock(dims[3] * 2, dims[3])
        self.up2 = nn.Conv2d(dims[3], dims[2] * 4, 1)
        self.dec2 = UNetBlock(dims[2] * 2, dims[2])
        self.up1 = nn.Conv2d(dims[2], dims[1] * 4, 1)
        self.dec1 = UNetBlock(dims[1] * 2, dims[1])
        self.up0 = nn.Conv2d(dims[1], dims[0] * 4, 1)
        self.dec0 = nn.Conv2d(dims[0] * 2, dims[0], 3, 1, 1)

        self.out_conv = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x, cond_bottleneck):
        # Encoder
        h0 = self.enc0(x)
        h1 = self.enc1(self.down0(F.pixel_unshuffle(h0, 2)))
        h2 = self.enc2(self.down1(F.pixel_unshuffle(h1, 2)))
        h3 = self.enc3(self.down2(F.pixel_unshuffle(h2, 2)))

        # Bottleneck @ H/16
        hb = self.down3(F.pixel_unshuffle(h3, 2))

        # Safety catch: Ensure transformer condition matches UNet bottleneck resolution
        if hb.shape[2:] != cond_bottleneck.shape[2:]:
            cond_bottleneck = F.interpolate(cond_bottleneck, size=hb.shape[2:], mode='nearest')

        # Efficient Concat
        h = self.mid(torch.cat([hb, cond_bottleneck], dim=1))

        # Decoder
        h = self.dec3(torch.cat([F.pixel_shuffle(self.up3(h), 2), h3], dim=1))
        h = self.dec2(torch.cat([F.pixel_shuffle(self.up2(h), 2), h2], dim=1))
        h = self.dec1(torch.cat([F.pixel_shuffle(self.up1(h), 2), h1], dim=1))
        h = self.dec0(torch.cat([F.pixel_shuffle(self.up0(h), 2), h0], dim=1))

        return self.out_conv(h)


# ─── Main Model ─────────────────────────────────────────────────────────────


class TokenformerDiT(nn.Module):
    def __init__(self, in_channels=3, dim=768, depth=12, num_heads=12,
                 num_classes=12476, use_checkpoint=False,
                 encoder_depth=2, decoder_depth=2,
                 unet_base_dim=64, patch_size=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size

        self.patch_embed = nn.Linear(in_channels * patch_size ** 2, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = MeanPoolingEmbedder(num_classes, dim)

        # Stage 1: Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(encoder_depth)
        ])

        # Stage 2: Mid
        self.down_proj = nn.Linear(dim * 4, dim, bias=False)
        mid_depth = depth - encoder_depth - decoder_depth
        self.mid_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(mid_depth)
        ])

        # Stage 3: Decoder
        self.up_proj = nn.Linear(dim, dim * 4, bias=False)
        self.skip_fusion = nn.Linear(dim * 2, dim, bias=False)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(decoder_depth)
        ])

        self.final_norm = RMSNorm(dim)

        # Logvar head: 1 scalar per transformer token (= per 16x16 patch block)
        self.logvar_head = nn.Linear(dim, 1, bias=True)

        # CLS denoising projections (DINOv3 ViT-L dim = 1024)
        self.cls_in_proj = nn.Linear(1024, dim, bias=False)
        self.cls_out_proj = nn.Linear(dim, 1024, bias=False)

        # REPA projection head (mid-block features → DINO patch token space)
        self.repa_projector = REPAProjector(dim, 1024)

        # UNet predicts x0 only (no logvar)
        self.unet = SmallUNet(in_channels, dim, unet_base_dim)

    def _run_block(self, block, x, H, W, context):
        if self.use_checkpoint and self.training:
            return checkpoint(block, x, H, W, context, use_reentrant=False)
        return block(x, H, W, context)

    def forward(self, x_in, t, y_indices, y_offsets=None, cls_embed=None):
        B, C, H, W = x_in.shape
        p = self.patch_size

        if p > 1:
            x_patch = F.pixel_unshuffle(x_in, p)
        else:
            x_patch = x_in
        Hp, Wp = H // p, W // p

        x = x_patch.flatten(2).transpose(1, 2)
        x = self.patch_embed(x)

        # --- Handle Spatial Maps vs Global Timesteps cleanly ---
        if t.ndim == 4:
            # Expect block-level spatial timestep map at patch resolution (B, 1, Hp, Wp)
            # Flatten to match the patched sequence shape (B, N)
            t = t.flatten(2).transpose(1, 2).squeeze(-1)
            t_emb = self.t_embedder(t)  # Shape: (B, N, dim)
            x = x + t_emb  # Directly add spatial embeddings
        else:
            # Standard Global Scalar Timesteps (B,)
            t = t.reshape(B)
            t_emb = self.t_embedder(t)  # Shape: (B, dim)
            x = x + t_emb.unsqueeze(1)  # Broadcast to (B, N, dim)
        # -------------------------------------------------------

        context = self.y_embedder(y_indices, y_offsets)

        # Prepend CLS token to context for denoising
        if cls_embed is not None:
            cls_tok = self.cls_in_proj(cls_embed).unsqueeze(1)  # (B, 1, dim)
            context = torch.cat([cls_tok, context], dim=1)      # (B, 5, dim)

        for block in self.encoder_blocks:
            x, context = self._run_block(block, x, Hp, Wp, context)
        skip = x

        x = x.transpose(1, 2).view(B, self.dim, Hp, Wp)
        x = F.pixel_unshuffle(x, 2)
        Hm, Wm = Hp // 2, Wp // 2
        x = x.flatten(2).transpose(1, 2)
        x = self.down_proj(x)

        mid_point = len(self.mid_blocks) // 2
        repa_hidden = None
        for i, block in enumerate(self.mid_blocks):
            x, context = self._run_block(block, x, Hm, Wm, context)
            if self.training and i == mid_point:
                repa_hidden = x

        x = self.up_proj(x)
        x = x.transpose(1, 2).view(B, self.dim * 4, Hm, Wm)
        x = F.pixel_shuffle(x, 2)
        x = x.flatten(2).transpose(1, 2)

        x = self.skip_fusion(torch.cat([x, skip], dim=-1))

        for block in self.decoder_blocks:
            x, context = self._run_block(block, x, Hp, Wp, context)

        # Reshape to 2D at the patched resolution to feed UNet Bottleneck
        x = self.final_norm(x)

        # Logvar: 1 value per 16x16 block, predicted directly from transformer tokens
        logvar_tokens = self.logvar_head(x)  # (B, N, 1)
        logvar = logvar_tokens.transpose(1, 2).view(B, 1, Hp, Wp)  # (B, 1, Hp, Wp)

        cond_features = x.transpose(1, 2).view(B, self.dim, Hp, Wp)

        # UNet predicts x0 only
        x0_pred = self.unet(x_in, cond_features)

        # REPA + CLS outputs
        cls_pred = None
        repa_feat = None
        if cls_embed is not None:
            cls_pred = self.cls_out_proj(context[:, 0])  # (B, 1024)
        if self.training and repa_hidden is not None:
            repa_feat = self.repa_projector(repa_hidden, Hm, Wm)

        return x0_pred, logvar, repa_feat, cls_pred


class TagProcessor:
    def __init__(self, tags_file):
        with open(tags_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.num_classes = len(self.tags)

    def process_prompts(self, prompts, device, dropout_prob=0.0):
        indices = []
        offsets = [0]
        for p in prompts:
            if random.random() < dropout_prob:
                indices.append(self.num_classes)
            else:
                tags = p.split()
                count = 0
                for t in tags:
                    if t in self.tag_to_idx:
                        indices.append(self.tag_to_idx[t])
                        count += 1
                if count == 0:
                    indices.append(self.num_classes)
            offsets.append(len(indices))

        indices = torch.tensor(indices, dtype=torch.long, device=device)
        offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return indices, offsets


@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=50, guidance_scale=1.5, noise=None, cfg_scale=0, **kwargs):
    """Standard global-timestep Flow Matching sampling."""
    in_channels = model.in_channels if hasattr(model, 'in_channels') else (
        model.module.in_channels if hasattr(model, 'module') else 128)
    model.eval()

    if isinstance(latent_size, (tuple, list)):
        H, W = latent_size
    else:
        H = W = latent_size

    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        x = torch.randn(batch_size, in_channels, H, W, device=device)

    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)
    null_prompts = [""] * batch_size
    y_null_indices, y_null_offsets = tag_processor.process_prompts(null_prompts, device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    x = x.to(torch.bfloat16)
    cls_state = torch.randn(batch_size, 1024, device=device, dtype=x.dtype)

    for i in tqdm(range(steps), desc="Euler Sampling"):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size,), t_curr.item(), device=device, dtype=x.dtype)

        x0_cond, _, _, cls_pred_cond = model(x, t_vec, y_indices, y_offsets, cls_embed=cls_state)
        x0_uncond, _, _, cls_pred_uncond = model(x, t_vec, y_null_indices, y_null_offsets, cls_embed=cls_state)

        x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

        t_safe = t_curr.clamp(min=0.05)
        v = (x - x0_guided) / t_safe
        x = x + dt * v

        # Denoise CLS token alongside
        cls_pred_guided = cls_pred_uncond + guidance_scale * (cls_pred_cond - cls_pred_uncond)
        v_cls = (cls_state - cls_pred_guided) / t_safe
        cls_state = cls_state + dt * v_cls

    model.train()
    return x


@torch.no_grad()
def sample_lookahead(model, tag_processor, latent_size, batch_size, prompts, device,
                     steps=250, guidance_scale=1.5, alpha=1.5, percentile=0.4,
                     noise=None, **kwargs):
    """
    Look-ahead Sampling (Algorithm S1 from Patch Forcing).
    Advances confident patches to provide context for uncertain patches.
    Operates with block-level (patch_size x patch_size) timestep maps.
    """
    in_channels = model.in_channels if hasattr(model, 'in_channels') else (
        model.module.in_channels if hasattr(model, 'module') else 128)
    p = model.patch_size if hasattr(model, 'patch_size') else (
        model.module.patch_size if hasattr(model, 'module') else 16)
    model.eval()

    if isinstance(latent_size, (tuple, list)):
        H, W = latent_size
    else:
        H = W = latent_size
    Hp, Wp = H // p, W // p

    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        x = torch.randn(batch_size, in_channels, H, W, device=device)

    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)
    null_prompts = [""] * batch_size
    y_null_indices, y_null_offsets = tag_processor.process_prompts(null_prompts, device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    x = x.to(torch.bfloat16)
    cls_state = torch.randn(batch_size, 1024, device=device, dtype=x.dtype)

    for i in tqdm(range(steps), desc="Look-ahead Sampling"):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr  # dt is negative

        # Block-level uniform timestep map (B, 1, Hp, Wp)
        t_vec = torch.full((batch_size, 1, Hp, Wp), t_curr.item(), device=device, dtype=x.dtype)

        # 1. Predict x0 and uncertainty (logvar) at current step
        x0_cond, logvar, _, cls_pred_cond = model(x, t_vec, y_indices, y_offsets, cls_embed=cls_state)
        x0_uncond, _, _, cls_pred_uncond = model(x, t_vec, y_null_indices, y_null_offsets, cls_embed=cls_state)

        x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

        t_safe = t_curr.clamp(min=0.05)
        v_t = (x - x0_guided) / t_safe

        # 2. Adaptive thresholding at block level
        # logvar is already (B, 1, Hp, Wp) at block resolution
        uc_flat = logvar.view(batch_size, -1)
        tau_p = torch.quantile(uc_flat.float(), percentile, dim=1).view(batch_size, 1, 1, 1).to(x.dtype)

        # 3. Create block-level confidence masks and expand to pixel level
        M_conf_block = (logvar <= tau_p).to(x.dtype)  # (B, 1, Hp, Wp)
        M_unc_block = 1.0 - M_conf_block
        M_conf_pixel = M_conf_block.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        M_unc_pixel = 1.0 - M_conf_pixel

        # 4. Context Look-ahead time (stepping closer to data/0.0)
        t_ctx = t_curr / alpha

        # 5. One-step look-ahead for confident patches
        x_ctx = x + (t_ctx - t_curr) * v_t

        # 6. Combine states (pixel level)
        x_tilde = M_conf_pixel * x_ctx + M_unc_pixel * x

        # 7. Block-level spatially mixed timestep for re-evaluation
        t_tilde_block = M_conf_block * t_ctx + M_unc_block * t_curr  # (B, 1, Hp, Wp)

        x0_cond_ctx, _, _, _ = model(x_tilde, t_tilde_block, y_indices, y_offsets, cls_embed=cls_state)
        x0_uncond_ctx, _, _, _ = model(x_tilde, t_tilde_block, y_null_indices, y_null_offsets, cls_embed=cls_state)

        x0_guided_ctx = x0_uncond_ctx + guidance_scale * (x0_cond_ctx - x0_uncond_ctx)

        # Expand block timestep to pixel level for velocity division
        t_tilde_pixel = t_tilde_block.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        t_tilde_safe = t_tilde_pixel.clamp(min=0.05)
        v_ctx = (x_tilde - x0_guided_ctx) / t_tilde_safe

        # 8. Combine final velocity (pixel level)
        v_final = M_unc_pixel * v_ctx + M_conf_pixel * v_t

        # 9. Step forward
        x = x + dt * v_final

        # Denoise CLS token alongside (global timestep, not spatial)
        cls_pred_guided = cls_pred_uncond + guidance_scale * (cls_pred_cond - cls_pred_uncond)
        v_cls = (cls_state - cls_pred_guided) / t_safe
        cls_state = cls_state + dt * v_cls

    model.train()
    return x