import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class GGRoPE2d(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
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

        H, W = image_size
        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_HW = torch.linspace(-xlim, xlim, W).reshape(1, W).expand(H, W)
        y_HW = torch.linspace(-ylim, ylim, H).reshape(H, 1).expand(H, W)
        positions_HW112 = torch.stack((x_HW, y_HW), dim=-1).reshape(H, W, 1, 1, 2)

        theta_HWhF = (freqs_hF2 * positions_HW112).sum(dim=-1)
        # Register flattened buffers for easy indexing (S, h, F)
        self.register_buffer("cos_Sf", torch.cos(theta_HWhF).reshape(-1, n_heads, n_freqs))
        self.register_buffer("sin_Sf", torch.sin(theta_HWhF).reshape(-1, n_heads, n_freqs))

    def forward(self, x: torch.Tensor, indices: torch.Tensor = None) -> torch.Tensor:
        # x shape: (B, h, N, d)
        B, h, N, d = x.shape
        F_dim = d // 2
        
        cos = self.cos_Sf # (S_orig, h, F)
        sin = self.sin_Sf
        
        if indices is not None:
            cos = cos[indices] # (N_kept, h, F)
            sin = sin[indices]
            
        # Reshape for broadcasting: (1, h, N, F)
        cos = cos.permute(1, 0, 2).unsqueeze(0)
        sin = sin.permute(1, 0, 2).unsqueeze(0)
        
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

    def forward(self, x, mask=None, rope=None, indices=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to spatial tokens only (indices 6 onwards)
        if rope is not None:
            q_prefix = q[:, :, :6, :]
            q_spatial = q[:, :, 6:, :]
            k_prefix = k[:, :, :6, :]
            k_spatial = k[:, :, 6:, :]
            
            q_spatial = rope(q_spatial, indices=indices)
            k_spatial = rope(k_spatial, indices=indices)
            
            q = torch.cat([q_prefix, q_spatial], dim=2)
            k = torch.cat([k_prefix, k_spatial], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
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

    def forward(self, x, rope=None, indices=None):
        x = x + self.attn(self.norm1(x), rope=rope, indices=indices)
        x = x + self.mlp(self.norm2(x))
        return x

class DiTSkip(nn.Module):
    def __init__(self, in_channels=32, dim=768, depth=12, num_heads=12, num_classes=12476, num_registers=4, latent_size=32, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Patch Embed
        self.patch_embed = nn.Linear(in_channels, dim)
        
        # Condition Embeds
        self.t_embedder = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.y_embedder = nn.EmbeddingBag(num_classes + 1, dim, mode='mean')
        
        # Register Tokens
        self.registers = nn.Parameter(torch.randn(1, num_registers, dim))
        
        # RoPE
        self.rope = GGRoPE2d(
            image_size=(latent_size, latent_size),
            n_heads=num_heads,
            head_dim=dim // num_heads,
            min_freq=1.0,
            max_freq=100.0
        )
        
        # Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads) for _ in range(depth)
        ])
        
        # Fusion Layer
        self.fuse = nn.Linear(dim * 2, dim, bias=False)
        # Learnable padding token for the 75% dropped spots
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.final_norm = RMSNorm(dim, elementwise_affine=False)
        self.final_proj = nn.Linear(dim, in_channels)

    def forward(self, x, t, y_indices, y_offsets=None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.patch_embed(x)
        
        # Condition tokens
        t_token = self.t_embedder(t.unsqueeze(-1)).unsqueeze(1)
        y_token = self.y_embedder(y_indices, y_offsets).unsqueeze(1)
        reg_tokens = self.registers.expand(B, -1, -1)
        
        # Prefix = Time(1) + Cond(1) + Registers(4) = 6 tokens
        x = torch.cat([t_token, y_token, reg_tokens, x], dim=1)
        
        full_res_skip = None
        keep_indices = None
        
        for i, block in enumerate(self.blocks):
            if i == 4:
                full_res_skip = x 
                prefix = x[:, :6, :]
                spatial = x[:, 6:, :]
                S = spatial.shape[1]
                
                # Random Sample Drop (75% drop, 25% keep)
                num_keep = max(1, S // 4)
                keep_indices = torch.randperm(S, device=x.device)[:num_keep]
                
                x = torch.cat([prefix, spatial[:, keep_indices, :]], dim=1)
            
            # Use current keep_indices for RoPE inside blocks
            # Before layer 4: keep_indices is None (Full 1024)
            # Layers 4 to N-2: keep_indices is num_keep (256)
            # After layer N-2: keep_indices is None (Full 1024) again
            
            if i == (self.depth - 2) and full_res_skip is not None:
                prefix = x[:, :6, :]
                dropped_spatial = x[:, 6:, :]
                
                # Reconstruct
                S_orig = full_res_skip.shape[1] - 6
                full_spatial = self.mask_token.expand(B, S_orig, -1).clone()
                full_spatial[:, keep_indices, :] = dropped_spatial
                
                x = torch.cat([prefix, full_spatial], dim=1)
                x = self.fuse(torch.cat([x, full_res_skip], dim=-1))
                # Reset keep_indices to None for final layers
                keep_indices = None

            # Pass rope and current keep_indices
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, self.rope, keep_indices, use_reentrant=False)
            else:
                x = block(x, rope=self.rope, indices=keep_indices)

        x = self.final_norm(x)
        x = self.final_proj(x)
        
        # Remove prefix tokens
        x = x[:, 6:, :]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x

@torch.no_grad()
def sample_flow(model, tag_processor, image_size, batch_size, prompts, device,
                steps=50, cfg_scale=1.4, noise=None):
    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        x = torch.randn(batch_size, model.final_proj.out_features, image_size, image_size, device=device)

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
