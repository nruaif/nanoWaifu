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
        x = self.mlp(x)
        return x.view(x.shape[0], 4, -1)  # (B, 4, dim)


class Pattention(nn.Module):
    """
    Token-Parameter Attention (Pattention) Layer optimized for PyTorch.
    This replaces explicit K_P/V_P tensors with two bias-free Linear layers.
    """

    def __init__(self, in_dim: int, out_dim: int, num_pairs: int):
        super().__init__()
        self.num_pairs = num_pairs

        # Layer 1 acts as K_P: Projects from input dimension to the number of parameter pairs (slots)
        self.k_proj = nn.Linear(in_dim, num_pairs, bias=False)

        # Layer 2 acts as V_P: Projects from the parameter slots back to the output dimension
        self.v_proj = nn.Linear(num_pairs, out_dim, bias=False)

        # Initialize weights to match the paper's scaling
        nn.init.normal_(self.k_proj.weight, std=in_dim ** -0.5)
        nn.init.normal_(self.v_proj.weight, std=out_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Similarity Scores: X @ K_P^T
        A = self.k_proj(x)  # Shape: (B, N, num_pairs)

        # 2. Modified Softmax: L2 norm + GeLU
        A_norm = F.normalize(A, p=2, dim=-1)
        S = F.gelu(A_norm * math.sqrt(self.num_pairs))

        # 3. Weighted Sum: S @ V_P
        O = self.v_proj(S)  # Shape: (B, N, out_dim)

        return O


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

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, h, N, d = x.shape
        F_dim = d // 2

        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_grid = torch.linspace(-xlim, xlim, W, device=x.device, dtype=x.dtype)
        y_grid = torch.linspace(-ylim, ylim, H, device=x.device, dtype=x.dtype)

        y_HW, x_HW = torch.meshgrid(y_grid, x_grid, indexing='ij')
        positions_HW2 = torch.stack((x_HW, y_HW), dim=-1).reshape(H * W, 1, 1, 2)

        theta = (self.freqs_hF2 * positions_HW2).sum(dim=-1)

        cos = torch.cos(theta).permute(1, 0, 2).unsqueeze(0)
        sin = torch.sin(theta).permute(1, 0, 2).unsqueeze(0)

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
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), weight=self.weight, eps=self.eps)


class TokenformerAttention(nn.Module):
    def __init__(self, dim, num_heads=12, attn_kv_pairs=576, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Replaced standard linear QKV with Pattention layers
        self.q_proj = Pattention(dim, dim, attn_kv_pairs)
        self.k_proj = Pattention(dim, dim, attn_kv_pairs)
        self.v_proj = Pattention(dim, dim, attn_kv_pairs)
        self.o_proj = Pattention(dim, dim, attn_kv_pairs)

        self.qk_norm = qk_norm
        if qk_norm:
            # Paper explicitly removes parameter weights from Norm to facilitate scaling
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x, H, W, rope=None):
        B, N, C = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope is not None:
            # Dynamically slice to isolate spatial tokens from prefixes (t_token + y_tokens)
            prefix_len = N - (H * W)
            q_prefix, q_spatial = q[:, :, :prefix_len, :], q[:, :, prefix_len:, :]
            k_prefix, k_spatial = k[:, :, :prefix_len, :], k[:, :, prefix_len:, :]

            q_spatial = rope(q_spatial, H, W)
            k_spatial = rope(k_spatial, H, W)

            q = torch.cat([q_prefix, q_spatial], dim=2)
            k = torch.cat([k_prefix, k_spatial], dim=2)

        x_att = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False
        )

        x_att = x_att.transpose(1, 2).reshape(B, N, C)
        return self.o_proj(x_att)


class TokenformerBlock(nn.Module):
    def __init__(self, dim, num_heads, attn_kv_pairs, ffn_kv_pairs):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = TokenformerAttention(dim, num_heads, attn_kv_pairs)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)

        # Tokenformer uses a single Pattention layer for the entire FFN block
        self.ffn = Pattention(dim, dim, ffn_kv_pairs)

    def forward(self, x, H, W, rope=None):
        x = x + self.attn(self.norm1(x), H, W, rope=rope)
        x = x + self.ffn(self.norm2(x))
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


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU(approximate="tanh")
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + shortcut


class TokenformerDiT(nn.Module):
    """
    Tokenformer Architecture (Default: 124M specs)
    """

    def __init__(self, in_channels=32, dim=768, depth=12, num_heads=12,
                 attn_kv_pairs=576, ffn_kv_pairs=2304, num_classes=12476,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.patch_embed = nn.Linear(in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = MeanPoolingEmbedder(num_classes, dim)

        self.rope = GGRoPE2d(
            n_heads=num_heads,
            head_dim=dim // num_heads,
            min_freq=1.0,
            max_freq=100.0
        )

        self.blocks = nn.ModuleList([
            TokenformerBlock(dim, num_heads, attn_kv_pairs, ffn_kv_pairs)
            for _ in range(depth)
        ])

        self.final_norm = RMSNorm(dim, elementwise_affine=False)
        self.final_proj = nn.Linear(dim, in_channels)

        self.conv_mix = ConvNeXtBlock(in_channels * 2)
        self.final_out_proj = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x_in, t, y_indices, y_offsets=None, return_features=False):
        B, C, H, W = x_in.shape
        x = x_in.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.patch_embed(x)

        # Prefix is now precisely 5 tokens: 1 for time, 4 for condition
        t_token = self.t_embedder(t).unsqueeze(1)
        y_token = self.y_embedder(y_indices, y_offsets)

        x = torch.cat([t_token, y_token, x], dim=1)

        # Removed internal DiT Skip Logic. Straight sequential processing.
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, H, W, self.rope, use_reentrant=False)
            else:
                x = block(x, H, W, rope=self.rope)

        feat = x
        x = self.final_norm(x)
        x = self.final_proj(x)

        # Remove the 5 prefix tokens
        x = x[:, 5:, :].reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Standard UNet-style end skip connection
        x = torch.cat([x, x_in], dim=1)
        x = self.conv_mix(x)
        x = self.final_out_proj(x)

        if not return_features:
            return x
        else:
            return x, feat


@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=50, noise=None):
    """
    Simplified sample_flow (removed CFG dropping logic as requested).
    """
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
    print("Initializing TokenformerDiT on CPU...")
    device = torch.device("cpu")

    # Initialize the model with the default 124M specs outlined in the paper
    model = TokenformerDiT(
        in_channels=32,
        dim=768,
        depth=12,
        num_heads=12,
        attn_kv_pairs=576,
        ffn_kv_pairs=2304,
        num_classes=12476
    ).to(device)

    # 1. Count and format parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params / 1e6:.2f} M")

    # 2. Create dummy inputs
    batch_size = 2
    latent_size = 16  # Represents H and W

    x_in = torch.randn(batch_size, 32, latent_size, latent_size, device=device)
    t = torch.rand(batch_size, device=device)

    # Dummy tags for the MeanPoolingEmbedder
    # Let's assume each item in the batch has exactly 3 tags
    y_indices = torch.randint(0, 12476, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)

    print("\nRunning forward pass...")

    # 3. Run the forward pass
    output = model(x_in, t, y_indices, y_offsets)

    # 4. Verification
    print(f"Input shape:  {x_in.shape}")
    print(f"Output shape: {output.shape}")

    assert x_in.shape == output.shape, "Error: Output shape does not match input shape!"
    print("✅ Forward pass completed successfully!")