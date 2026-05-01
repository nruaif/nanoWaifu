import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
import random


class MeanPoolingEmbedder(nn.Module):
    def __init__(self, num_classes: int, dim: int, num_context_tokens: int = 4):
        super().__init__()
        self.num_context_tokens = num_context_tokens
        self.embed = nn.EmbeddingBag(num_classes + 1, dim * num_context_tokens, mode='mean')

    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor) -> torch.Tensor:
        x = self.embed(y_indices, y_offsets)
        return x.view(x.shape[0], self.num_context_tokens, -1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.act = nn.GELU(approximate="tanh")
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Zero init output projection
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)
        if H is not None and W is not None:
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.dwconv(x)
            x = x.flatten(2).transpose(1, 2)
        return self.fc2(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

        # Zero init output projection
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N_seq, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N_seq, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x_att = F.scaled_dot_product_attention(q, k, v)
        x_att = x_att.transpose(1, 2).reshape(B, N_seq, C)
        return self.proj(x_att)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads)
        self.norm3 = RMSNorm(dim)
        self.mlp = Mlp(dim, dim * 4, dim)

        # Norm for context tokens before they hit the shared MLP
        self.context_norm = RMSNorm(dim)

    def forward(self, x, H, W, context=None, shared_context_mlp=None):
        B, N, C = x.shape

        # 1. Sequence Assembly for Attention
        num_context_tokens = 0
        if context is not None:
            x_seq = torch.cat([context, x], dim=1)
            num_context_tokens = context.shape[1]
        else:
            x_seq = x

        # 2. Joint Self-Attention
        x_norm1 = self.norm1(x_seq)
        attn_out = self.self_attn(x_norm1)
        x_seq = x_seq + attn_out

        # 3. SPLIT the sequence BEFORE the small MLP
        if context is not None:
            context_attn = x_seq[:, :num_context_tokens, :]
            x_attn = x_seq[:, num_context_tokens:, :]
        else:
            x_attn = x_seq
            context_attn = None

        # 4. Image tokens process through block's small MLP (with DW conv)
        x_out = x_attn + self.mlp(self.norm3(x_attn), H, W)

        # 5. Context tokens bypass small MLP, process through Shared Large MLP
        if context_attn is not None and shared_context_mlp is not None:
            c_normed = self.context_norm(context_attn)
            c_flat = c_normed.view(B, -1)  # (B, num_context_tokens * dim)
            c_processed = shared_context_mlp(c_flat)  # (B, num_context_tokens * dim)
            c_reshaped = c_processed.view(B, num_context_tokens, C)
            context_out = context_attn + c_reshaped
        else:
            context_out = context_attn

        return x_out, context_out


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
        t = t * 1000.0
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        if t.dim() == 1:
            emb = t.float()[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).to(self.mlp[0].weight.dtype)
            return self.mlp(emb)
        elif t.dim() == 2:
            emb = t.float().unsqueeze(-1) * emb.view(1, 1, -1)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1).to(self.mlp[0].weight.dtype)
            return self.mlp(emb)


class TokenformerDiT(nn.Module):
    """
    DiT architecture with pixel-shuffle downsampling in the middle blocks.
    Takes pre-patchified input (B, C*p^2, Hp, Wp).
    Conditions via concatenated tokens in self-attention.
    Utilizes a Shared MLP for context tokens to build global representations.
    """

    def __init__(self, in_channels=3, dim=768, depth=12, num_heads=12,
                 num_classes=12476, use_checkpoint=False, num_context_tokens=6,
                 encoder_depth=2, decoder_depth=2, patch_size=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_context_tokens = num_context_tokens

        # Input: pre-patchified (B, C*p^2, Hp, Wp)
        patchified_channels = in_channels * patch_size * patch_size
        self.patchified_channels = patchified_channels

        self.patch_embed_conv = nn.Conv2d(patchified_channels, dim, kernel_size=3, padding=1)

        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = MeanPoolingEmbedder(num_classes, dim, num_context_tokens)

        # Global shared MLP for context tokens across all layers
        self.shared_context_mlp = Mlp(
            in_features=dim * num_context_tokens,
            hidden_features=dim * num_context_tokens * 3,
            out_features=dim * num_context_tokens
        )

        self.num_encoder_blocks = encoder_depth
        self.num_decoder_blocks = decoder_depth
        self.num_middle_blocks = depth - self.num_encoder_blocks - self.num_decoder_blocks

        self.encoder_blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(self.num_encoder_blocks)])
        self.middle_blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(self.num_middle_blocks)])
        self.decoder_blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(self.num_decoder_blocks)])

        # Pixel-shuffle downsampling/upsampling (factor 2) around middle blocks
        self.downsample_proj = nn.Linear(dim * 4, dim)
        self.upsample_proj = nn.Linear(dim, dim * 4)

        # Skip connection fusion (encoder → decoder)
        self.fusion = nn.Linear(dim * 2, dim)

        self.final_norm = RMSNorm(dim)
        # Conv2d with kernel 3 to project back to patchified pixel space + logvar
        self.final_proj = nn.Conv2d(dim, patchified_channels + 1, kernel_size=3, padding=1)


    def forward(self, x_in, t, y_indices, y_offsets=None):
        """
        Args:
            x_in: Pre-patchified input (B, C*p^2, Hp, Wp)
            t: Timesteps - scalar (B,) or block-level (B, 1, Hp, Wp)
            y_indices: Tag indices for conditioning
            y_offsets: Offsets for EmbeddingBag
        Returns:
            u_pred: Predicted x0 in patchified space (B, C*p^2, Hp, Wp)
            logvar_theta: Log-variance at block level (B, 1, Hp, Wp)
        """
        B, _, Hp, Wp = x_in.shape
        N = Hp * Wp

        x_in = self.patch_embed_conv(x_in)
        x = x_in.flatten(2).transpose(1, 2)  # (B, N, dim)

        # Timestep embedding
        if t.dim() == 1:
            t_flat = t.unsqueeze(1).expand(-1, N)
        elif t.dim() == 4:
            # (B, 1, Hp, Wp) → (B, Hp*Wp)
            t_flat = t.flatten(1)
        else:
            t_flat = t
        t_embeds = self.t_embedder(t_flat)
        x = x + t_embeds

        # Class conditioning tokens
        y_tokens = self.y_embedder(y_indices, y_offsets)
        context = y_tokens

        # ─── Encoder blocks (full resolution: Hp × Wp) ─────────────────
        for block in self.encoder_blocks:
            if self.use_checkpoint and self.training:
                x, context = checkpoint(block, x, Hp, Wp, context, self.shared_context_mlp,
                                        use_reentrant=False)
            else:
                x, context = block(x, Hp, Wp, context, self.shared_context_mlp)

        f_t = x  # Save encoder output for skip connection

        # ─── Downsample via pixel_unshuffle(2) ──────────────────────────
        # (B, N, dim) → (B, dim, Hp, Wp) → unshuffle → (B, dim*4, Hp/2, Wp/2) → proj → (B, dim, Hp/2, Wp/2)
        x_spatial = x.transpose(1, 2).reshape(B, self.dim, Hp, Wp)
        x_down = F.pixel_unshuffle(x_spatial, 2)  # (B, dim*4, Hp/2, Wp/2)
        Hp_mid, Wp_mid = Hp // 2, Wp // 2
        x_middle = x_down.flatten(2).transpose(1, 2)  # (B, Hp/2*Wp/2, dim*4)
        x_middle = self.downsample_proj(x_middle)  # (B, Hp/2*Wp/2, dim)

        # ─── Middle blocks (half resolution: Hp/2 × Wp/2) ──────────────
        for block in self.middle_blocks:
            if self.use_checkpoint and self.training:
                x_middle, context = checkpoint(block, x_middle, Hp_mid, Wp_mid, context, self.shared_context_mlp,
                                               use_reentrant=False)
            else:
                x_middle, context = block(x_middle, Hp_mid, Wp_mid, context, self.shared_context_mlp)

        # ─── Upsample via pixel_shuffle(2) ──────────────────────────────
        # (B, Hp/2*Wp/2, dim) → proj → (B, Hp/2*Wp/2, dim*4) → (B, dim*4, Hp/2, Wp/2) → shuffle → (B, dim, Hp, Wp)
        x_up = self.upsample_proj(x_middle)  # (B, Hp/2*Wp/2, dim*4)
        x_up = x_up.transpose(1, 2).reshape(B, self.dim * 4, Hp_mid, Wp_mid)
        x_up = F.pixel_shuffle(x_up, 2)  # (B, dim, Hp, Wp)
        g = x_up.flatten(2).transpose(1, 2)  # (B, N, dim)

        # ─── Fusion (skip connection) ───────────────────────────────────
        h_t = self.fusion(torch.cat([f_t, g], dim=-1))

        # ─── Decoder blocks (full resolution: Hp × Wp) ─────────────────
        for block in self.decoder_blocks:
            if self.use_checkpoint and self.training:
                h_t, context = checkpoint(block, h_t, Hp, Wp, context, self.shared_context_mlp,
                                          use_reentrant=False)
            else:
                h_t, context = block(h_t, Hp, Wp, context, self.shared_context_mlp)

        # ─── Final projection via Conv2d ────────────────────────────────
        x_norm = self.final_norm(h_t)
        x_spatial = x_norm.transpose(1, 2).reshape(B, self.dim, Hp, Wp)
        x_out = self.final_proj(x_spatial)  # (B, C*p^2 + 1, Hp, Wp)

        x0_pred = x_out[:, :self.patchified_channels, :, :]
        logvar_theta = x_out[:, self.patchified_channels:, :, :]

        return x0_pred, logvar_theta


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
                steps=50, guidance_scale=2, noise=None, cfg_scale=0,
                sampler_type="euler", p_percentile=0.4, alpha=2.0):
    """
    Sampling operates in patchified space.
    latent_size: pixel-level (H, W) or int. Will be converted to patch-level internally.
    Returns: pixel-level output (B, C, H, W) after pixel_shuffle.
    """
    m = model.module if hasattr(model, 'module') else model
    in_channels = m.in_channels
    patch_size = m.patch_size
    patchified_channels = m.patchified_channels

    model.eval()
    if isinstance(latent_size, (tuple, list)):
        H, W = latent_size
    else:
        H = W = latent_size

    Hp, Wp = H // patch_size, W // patch_size

    # Work in patchified space: (B, C*p^2, Hp, Wp)
    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        x = torch.randn(batch_size, patchified_channels, Hp, Wp, device=device)

    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)
    null_prompts = [""] * batch_size
    y_null_indices, y_null_offsets = tag_processor.process_prompts(null_prompts, device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    x = x.to(torch.bfloat16)
    for i in tqdm(range(steps)):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size, Hp * Wp), t_curr.item(), device=device, dtype=x.dtype)
        t_safe = max(t_curr.item(), 0.05)

        if sampler_type == "euler":
            x0_cond, logvar_cond = model(x, t_vec, y_indices, y_offsets)
            v_cond = (x - x0_cond) / t_safe

            x0_uncond, _ = model(x, t_vec, y_null_indices, y_null_offsets)
            v_uncond = (x - x0_uncond) / t_safe

            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            x = x + dt * v

        elif sampler_type == "look-ahead":
            x0_cond, logvar_cond = model(x, t_vec, y_indices, y_offsets)
            v_cond = (x - x0_cond) / t_safe

            x0_uncond, _ = model(x, t_vec, y_null_indices, y_null_offsets)
            v_uncond = (x - x0_uncond) / t_safe

            v = v_uncond + guidance_scale * (v_cond - v_uncond)

            uc = logvar_cond
            uc_flat = uc.view(batch_size, -1)
            tau_p = torch.quantile(uc_flat.float(), p_percentile, dim=1, keepdim=True).to(x.dtype)
            tau_p = tau_p.view(batch_size, 1, 1, 1)

            M_conf = (uc <= tau_p).to(x.dtype)
            M_unc = 1.0 - M_conf

            t_ctx_val = max(t_curr.item() + alpha * dt.item(), 0.0)

            x_ctx = x + (t_ctx_val - t_curr.item()) * v
            x_tilde = M_conf * x_ctx + M_unc * x

            t_tilde_vec = M_conf.view(batch_size, -1) * t_ctx_val + M_unc.view(batch_size, -1) * t_curr.item()
            t_tilde_safe = t_tilde_vec.clamp(min=0.05).view(batch_size, 1, Hp, Wp)

            x0_ctx_cond, _ = model(x_tilde, t_tilde_vec, y_indices, y_offsets)
            v_ctx_cond = (x_tilde - x0_ctx_cond) / t_tilde_safe

            x0_ctx_uncond, _ = model(x_tilde, t_tilde_vec, y_null_indices, y_null_offsets)
            v_ctx_uncond = (x_tilde - x0_ctx_uncond) / t_tilde_safe

            v_ctx = v_ctx_uncond + guidance_scale * (v_ctx_cond - v_ctx_uncond)

            v_final = M_unc * v_ctx + M_conf * v
            x = x + dt * v_final

    # Unpatchify back to pixel space
    x_pixel = F.pixel_shuffle(x, patch_size)  # (B, C, H, W)

    model.train()
    return x_pixel


if __name__ == "__main__":
    print("Initializing DiT on CPU...")
    device = torch.device("cpu")

    model = TokenformerDiT(
        in_channels=3,
        dim=768,
        depth=16,
        num_heads=12,
        num_classes=12476,
        num_context_tokens=3,
        patch_size=16
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params / 1e6:.2f} M")

    batch_size = 2
    patch_size = 16
    Hp, Wp = 16, 16  # 256/16

    # Pre-patchified input
    x_in = torch.randn(batch_size, 3 * patch_size * patch_size, Hp, Wp, device=device)
    t = torch.rand(batch_size, device=device)

    y_indices = torch.randint(0, 12476, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)

    print("\nRunning forward pass (training)...")
    model.train()
    x0_pred, logvar = model(x_in, t, y_indices, y_offsets)

    print(f"Input shape:  {x_in.shape}")
    print(f"Output x0_pred shape: {x0_pred.shape}")
    print(f"Output logvar shape: {logvar.shape}")

    # Unpatchify
    x_pixel = F.pixel_shuffle(x0_pred, patch_size)
    print(f"Unpatchified output: {x_pixel.shape}")

    print("✅ Forward pass completed successfully!")