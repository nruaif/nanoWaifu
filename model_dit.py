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
            
        cos = torch.cos(theta).permute(1, 0, 2).unsqueeze(0) # (1, h, N, F)
        sin = torch.sin(theta).permute(1, 0, 2).unsqueeze(0) # (1, h, N, F)
        
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
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

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
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True)

    def forward(self, x, H, W, rope=None, indices=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

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

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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

class DiTSkip(nn.Module):
    def __init__(self, in_channels=32, dim=768, depth=12, num_heads=12, num_classes=12476, num_registers=4, use_checkpoint=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.patch_embed = nn.Linear(in_channels, dim)
        self.t_embedder = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.y_embedder = nn.EmbeddingBag(num_classes + 1, dim, mode='mean')
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

    def forward(self, x, t, y_indices, y_offsets=None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.patch_embed(x)
        
        t_token = self.t_embedder(t.unsqueeze(-1)).unsqueeze(1)
        y_token = self.y_embedder(y_indices, y_offsets).unsqueeze(1)
        reg_tokens = self.registers.expand(B, -1, -1)
        x = torch.cat([t_token, y_token, reg_tokens, x], dim=1)
        
        full_res_skip = None
        keep_indices = None
        
        for i, block in enumerate(self.blocks):
            if i == 4:
                full_res_skip = x 
                prefix = x[:, :6, :]
                spatial = x[:, 6:, :]
                S = spatial.shape[1]
                num_keep = max(1, S // 4)
                keep_indices = torch.randperm(S, device=x.device)[:num_keep]
                x = torch.cat([prefix, spatial[:, keep_indices, :]], dim=1)
            
            if i == (self.depth - 2) and full_res_skip is not None:
                prefix = x[:, :6, :]
                dropped_spatial = x[:, 6:, :]
                S_orig = full_res_skip.shape[1] - 6
                full_spatial = self.mask_token.expand(B, S_orig, -1).clone()
                full_spatial[:, keep_indices, :] = dropped_spatial
                x = torch.cat([prefix, full_spatial], dim=1)
                x = self.fuse(torch.cat([x, full_res_skip], dim=-1))
                keep_indices = None

            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, H, W, self.rope, keep_indices, use_reentrant=False)
            else:
                x = block(x, H, W, rope=self.rope, indices=keep_indices)

        x = self.final_norm(x)
        x = self.final_proj(x)
        x = x[:, 6:, :].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x

@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=50, cfg_scale=1.4, noise=None):
    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        # For sampling, we use latent_size directly as H and W
        x = torch.randn(batch_size, model.final_proj.out_features, latent_size, latent_size, device=device)

    y_indices, y_offsets = tag_processor.process_prompts(prompts, device)
    null_indices = torch.full((batch_size,), fill_value=tag_processor.num_classes, dtype=torch.long, device=device)
    null_offsets = torch.arange(batch_size, dtype=torch.long, device=device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i+1]
        t_vec = torch.full((batch_size,), t_curr, device=device)
        
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t_vec, t_vec], dim=0)
        y_idx_in = torch.cat([y_indices, null_indices], dim=0)
        y_off_in = torch.cat([y_offsets, null_offsets + y_indices.shape[0]], dim=0)
        
        v_pred = model(x_in, t_in, y_idx_in, y_off_in)
        v_cond, v_uncond = v_pred.chunk(2, dim=0)
        v_final = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        if t_curr > 0:
            x = (t_next / t_curr) * x + (1 - t_next / t_curr) * v_final
        else:
            x = v_final
            
    return x
