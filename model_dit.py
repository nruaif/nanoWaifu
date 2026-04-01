import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class MeanPoolingEmbedder(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        # Projects to dim * 4 so we can reshape into 4 distinct tokens
        self.embed = nn.EmbeddingBag(num_classes + 1, dim * 4, mode='mean')
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 16),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 16, dim * 4),
        )
    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor) -> torch.Tensor:
        """
        y_indices: (N,) concatenated indices
        y_offsets: (B,) start offsets into y_indices
        """
        x = self.embed(y_indices, y_offsets)  # (B, dim * 4)
        x = x + self.mlp(x)
        return x.view(x.shape[0], 4, -1)  # (B, 4, dim)


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
        self.register_buffer("freqs_hF2", freqs_hF2)

        # ADD A CACHE DICTIONARY
        self._cache = {}

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, h, N, d = x.shape

        # CHECK CACHE FIRST
        cache_key = (H, W, x.device, x.dtype)
        if cache_key not in self._cache:
            xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
            x_grid = torch.linspace(-xlim, xlim, W, device=x.device, dtype=x.dtype)
            y_grid = torch.linspace(-ylim, ylim, H, device=x.device, dtype=x.dtype)

            y_HW, x_HW = torch.meshgrid(y_grid, x_grid, indexing='ij')
            positions_HW2 = torch.stack((x_HW, y_HW), dim=-1).reshape(H * W, 1, 1, 2)

            theta = (self.freqs_hF2 * positions_HW2).sum(dim=-1)

            cos = torch.cos(theta).permute(1, 0, 2).unsqueeze(0)
            sin = torch.sin(theta).permute(1, 0, 2).unsqueeze(0)

            self._cache[cache_key] = (cos, sin)

        # RETRIEVE FROM CACHE
        cos, sin = self._cache[cache_key]

        x_fp32 = x.float()
        x1, x2 = x_fp32.chunk(2, dim=-1)

        x_out1 = x1 * cos - x2 * sin
        x_out2 = x1 * sin + x2 * cos

        output = torch.cat((x_out1, x_out2), dim=-1)
        return output.type_as(x)


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


class MlpDW(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

    def forward(self, x, H, W, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = rope(q, H, W)
        k = rope(k, H, W)

        x_att = F.scaled_dot_product_attention(q, k, v)
        x_att = x_att.transpose(1, 2).reshape(B, N, C)
        return self.proj(x_att)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(context_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape
        
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: t.view(B, M, self.num_heads, self.head_dim).transpose(1, 2), kv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x_att = F.scaled_dot_product_attention(q, k, v)
        x_att = x_att.transpose(1, 2).reshape(B, N, C)
        return self.proj(x_att)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, use_cross_attn=False, context_dim=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads)
        
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm2 = RMSNorm(dim)
            self.cross_attn = CrossAttention(dim, context_dim, num_heads)
            
        self.norm3 = RMSNorm(dim)
        self.mlp = MlpDW(dim, dim * 4, dim)

    def forward(self, x, rope, context=None):
        B, C, H, W = x.shape
        
        # Self attention
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm1 = self.norm1(x_flat)
        attn_out = self.self_attn(x_norm1, H, W, rope)
        x = x + attn_out.transpose(1, 2).reshape(B, C, H, W)
        
        # Cross attention
        if self.use_cross_attn:
            x_flat = x.flatten(2).transpose(1, 2)
            x_norm2 = self.norm2(x_flat)
            cross_out = self.cross_attn(x_norm2, context)
            x = x + cross_out.transpose(1, 2).reshape(B, C, H, W)
            
        # MLP
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm3 = self.norm3(x_flat).transpose(1, 2).reshape(B, C, H, W)
        mlp_out = self.mlp(x_norm3)
        x = x + mlp_out
        
        return x


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
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)


class TokenformerDiT(nn.Module):
    """
    DiT with UNet architecture (2-4-2 blocks configs).
    Conditions via cross attention at the mid stage, keeping the 2D feature map.
    Includes DW conv in the MLP block.
    """
    def __init__(self, in_channels=32, dim=768, depth=12, num_heads=12,
                 attn_kv_pairs=576, ffn_kv_pairs=2304, num_classes=12476,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=1)
        
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = MeanPoolingEmbedder(num_classes, dim)

        self.rope = GGRoPE2d(
            n_heads=num_heads,
            head_dim=dim // num_heads,
            min_freq=1.0,
            max_freq=100.0
        )

        # 2-4-2 blocks config
        # Down stage: 2 blocks
        self.down_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, use_cross_attn=False) for _ in range(2)
        ])
        
        self.downsample = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        
        # Mid stage: 4 blocks (Cross Attention injected here)
        self.mid_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, use_cross_attn=True, context_dim=dim) for _ in range(4)
        ])
        
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)
        
        # Up stage: 2 blocks
        self.up_projs = nn.ModuleList([
            nn.Conv2d(dim * 2, dim, kernel_size=1) for _ in range(2)
        ])
        self.up_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, use_cross_attn=False) for _ in range(2)
        ])

        self.final_norm = RMSNorm(dim)
        self.final_proj = nn.Conv2d(dim, in_channels, kernel_size=1)

        self.final_out_proj = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x_in, t, y_indices, y_offsets=None, return_features=False):
        B, C, H, W = x_in.shape
        x = self.patch_embed(x_in)
        
        # Prefix tokens for context (t + 4*y = 5 tokens)
        t_token = self.t_embedder(t).unsqueeze(1)
        y_tokens = self.y_embedder(y_indices, y_offsets)
        context = torch.cat([t_token, y_tokens], dim=1)
        
        skips = []
        
        # Down blocks
        for block in self.down_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, self.rope, None, use_reentrant=False)
            else:
                x = block(x, self.rope, None)
            skips.append(x)
            
        x = self.downsample(x)
        
        # Mid blocks
        for block in self.mid_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, self.rope, context, use_reentrant=False)
            else:
                x = block(x, self.rope, context)
                
        feat = x.flatten(2).transpose(1, 2)
        
        x = self.upsample(x)
        
        # Up blocks
        for proj, block, skip in zip(self.up_projs, self.up_blocks, reversed(skips)):
            x = torch.cat([x, skip], dim=1)
            x = proj(x)
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, self.rope, None, use_reentrant=False)
            else:
                x = block(x, self.rope, None)
                
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.final_norm(x_flat).transpose(1, 2).reshape(B, self.dim, H, W)
        x = self.final_proj(x_norm)

        # Standard UNet-style end skip connection
        x_out = torch.cat([x, x_in], dim=1)
        x_out = self.final_out_proj(x_out)

        if not return_features:
            return x_out
        else:
            return x_out, feat

@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=50, noise=None):
    """
    Simplified sample_flow (removed CFG dropping logic as requested).
    """
    in_channels = model.final_proj.out_channels
    model.eval()
    H = W = latent_size

    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        x = torch.randn(batch_size, in_channels, H, W, device=device)

    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size,), t_curr.item(), device=device)

        # Single conditional pass
        v_final = model(x, t_vec, y_indices, y_offsets)

        x = x + dt * v_final

    model.train()
    return x


if __name__ == "__main__":
    print("Initializing UNetDiT on CPU...")
    device = torch.device("cpu")

    model = TokenformerDiT(
        in_channels=32,
        dim=768,
        num_heads=12,
        num_classes=12476
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params / 1e6:.2f} M")

    batch_size = 2
    latent_size = 16  # Represents H and W

    x_in = torch.randn(batch_size, 32, latent_size, latent_size, device=device)
    t = torch.rand(batch_size, device=device)

    y_indices = torch.randint(0, 12476, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)

    print("\nRunning forward pass...")

    output = model(x_in, t, y_indices, y_offsets)

    print(f"Input shape:  {x_in.shape}")
    print(f"Output shape: {output.shape}")

    assert x_in.shape == output.shape, "Error: Output shape does not match input shape!"
    print("✅ Forward pass completed successfully!")