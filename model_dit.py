import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class GGRoPE2d(nn.Module):
    def __init__(
            self,
            n_heads: int,
            head_dim: int,
            min_freq: float,
            max_freq: float,
            p_zero_freqs: float = 0.0,
            direction_spacing: float = math.pi * (math.sqrt(5) - 1) / 2,
    ):
        super().__init__()
        assert head_dim % 2 == 0
        assert 0 <= p_zero_freqs <= 1
        self.n_heads = n_heads
        self.head_dim = head_dim
        n_freqs = head_dim // 2
        n_zero_freqs = round(p_zero_freqs * n_freqs)

        omega_F = torch.cat(
            (
                torch.zeros(n_zero_freqs),
                min_freq
                * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
            )
        )
        phi_hF = (
                torch.arange(n_heads * n_freqs).reshape(n_heads, n_freqs)
                * direction_spacing
        )
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
        freqs_hF2 = omega_F.unsqueeze(-1) * directions_hF2

        # Store as buffer to stay on correct device
        self.register_buffer("freqs_hF2", freqs_hF2)

    def forward(self, x: torch.Tensor, H: int, W: int, indices: torch.Tensor = None) -> torch.Tensor:
        # x shape: (B, h, N, d)
        B, h, N, d = x.shape
        F_dim = d // 2

        # Dynamically generate grid for current H, W
        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_grid = torch.linspace(-xlim, xlim, W, device=x.device, dtype=x.dtype)
        y_grid = torch.linspace(-ylim, ylim, H, device=x.device, dtype=x.dtype)

        y_HW, x_HW = torch.meshgrid(y_grid, x_grid, indexing='ij')
        positions_HW2 = torch.stack((x_HW, y_HW), dim=-1).reshape(H * W, 1, 1, 2)

        # theta shape: (S_orig, h, F)
        theta = (self.freqs_hF2 * positions_HW2).sum(dim=-1)

        if indices is not None:
            theta = theta[indices]

        cos = torch.cos(theta).permute(1, 0, 2).unsqueeze(0)  # (1, h, N, F)
        sin = torch.sin(theta).permute(1, 0, 2).unsqueeze(0)  # (1, h, N, F)

        x_fp32 = x.float()
        x1, x2 = x_fp32.chunk(2, dim=-1)

        x_out1 = x1 * cos - x2 * sin
        x_out2 = x1 * sin + x2 * cos

        output = torch.cat((x_out1, x_out2), dim=-1)
        return output.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), weight=None, eps=self.eps)


class QKNormAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

    def forward(self, x, H, W, rope=None, indices=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim) \
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope is not None:
            q_prefix = q[:, :, :6, :]
            q_spatial = q[:, :, 6:, :]
            k_prefix = k[:, :, :6, :]
            k_spatial = k[:, :, 6:, :]

            q_spatial = rope(q_spatial, H, W, indices=indices)
            k_spatial = rope(k_spatial, H, W, indices=indices)

            q = torch.cat([q_prefix, q_spatial], dim=2)
            k = torch.cat([k_prefix, k_spatial], dim=2)

        # 🔥 SDPA replaces everything below
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,  # set >0 if training with dropout
            is_causal=False
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = QKNormAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False)
        )

    def forward(self, x, H, W, rope=None, indices=None):
        x = x + self.attn(self.norm1(x), H, W, rope=rope, indices=indices)
        x = x + self.mlp(self.norm2(x))
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hidden_dim = hidden_dim

    def forward(self, t):
        # Scale t by 1000 to match standard diffusion frequencies
        t = t * 1000.0
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)


class ConvNeXtBlock(nn.Module):
    """Simple ConvNeXt-style block for final token mixing."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + shortcut


class DiTSkip(nn.Module):
    def __init__(self, in_channels=32, dim=768, depth=12, num_heads=12, num_classes=12476, num_registers=4,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.patch_embed = nn.Linear(in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = nn.EmbeddingBag(num_classes + 1, dim, mode='mean', )
        self.registers = nn.Parameter(torch.randn(1, num_registers, dim))

        self.rope = GGRoPE2d(
            n_heads=num_heads,
            head_dim=dim // num_heads,
            min_freq=1.0,
            max_freq=100.0
        )

        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(depth)])
        self.fuse = nn.Linear(dim * 2, dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.final_norm = RMSNorm(dim, elementwise_affine=False)
        self.final_proj = nn.Linear(dim, in_channels)

        self.conv_mix = ConvNeXtBlock(in_channels * 2)
        self.final_out_proj = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x_in, t, y_indices, y_offsets=None, drop_tokens=False):
        B, C, H, W = x_in.shape
        x = x_in.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.patch_embed(x)

        t_token = self.t_embedder(t).unsqueeze(1)
        y_token = self.y_embedder(y_indices, y_offsets).unsqueeze(1)
        reg_tokens = self.registers.expand(B, -1, -1)
        x = torch.cat([t_token, y_token, reg_tokens, x], dim=1)

        full_res_skip = None
        keep_indices = None

        for i, block in enumerate(self.blocks):
            if i == 2:
                full_res_skip = x

                # 🔥 Refactored dropping logic for CFG
                if drop_tokens:
                    do_drop = True  # Forced drop for unconditional CFG pass
                elif self.training:
                    do_drop = torch.rand(1, device=x.device).item() > 0.1  # Standard training drop
                else:
                    do_drop = False  # Default for conditional inference is no drop

                if do_drop:
                    prefix = x[:, :6, :]
                    spatial = x[:, 6:, :]
                    S = spatial.shape[1]
                    num_keep = max(1, S // 4)

                    keep_indices = torch.randperm(S, device=x.device)[:num_keep]
                    x = torch.cat([prefix, spatial[:, keep_indices, :]], dim=1)
                else:
                    keep_indices = None  # no dropping

            if i == (self.depth - 3) and full_res_skip is not None:
                prefix = x[:, :6, :]

                if keep_indices is not None:
                    # 🔥 Reconstruct if we actually dropped
                    dropped_spatial = x[:, 6:, :]
                    S_orig = full_res_skip.shape[1] - 6

                    full_spatial = self.mask_token.expand(B, S_orig, -1).clone()
                    full_spatial[:, keep_indices, :] = dropped_spatial
                else:
                    # 🔥 No drop → already full resolution
                    full_spatial = x[:, 6:, :]

                x = torch.cat([prefix, full_spatial], dim=1)
                x = self.fuse(torch.cat([x, full_res_skip], dim=-1))

                keep_indices = None  # reset

            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, H, W, self.rope, keep_indices, use_reentrant=False)
            else:
                x = block(x, H, W, rope=self.rope, indices=keep_indices)

        x = self.final_norm(x)
        x = self.final_proj(x)
        x = x[:, 6:, :].reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x = torch.cat([x, x_in], dim=1)
        x = self.conv_mix(x)

        x = self.final_out_proj(x)

        return x


@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=50, cfg_scale=1.4, noise=None):
    # --- Initial noise ---
    in_channels = model.final_proj.out_features
    model.eval()
    H = W = latent_size

    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        x = torch.randn(batch_size, in_channels, H, W, device=device)

    # --- Conditioning ---
    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)

    # ❌ Null embedding logic removed; CFG now relies purely on token dropping

    # --- Euler integration: t from 1 (noise) → 0 (data) ---
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)  # t[0]=1, t[-1]=0

    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size,), t_curr.item(), device=device)

        # 🔥 Two unbatched passes because sequence lengths diverge inside the blocks
        # Conditional pass (Full spatial tokens)
        v_cond = model(x, t_vec, y_indices, y_offsets, drop_tokens=False)

        # Unconditional pass (Dropped spatial tokens)
        v_uncond = model(x, t_vec, y_indices, y_offsets, drop_tokens=True)

        # CFG: interpolate/extrapolate between unconditional and conditional
        v_final = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler step
        x = x + dt * v_final
    model.train()

    return x
