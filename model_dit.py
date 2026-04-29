import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
import random


# ─── Core Building Blocks ───────────────────────────────────────────────────


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
        """t: (B,) scalar timesteps."""
        t = t * 1000.0
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


# ─── Transformer Components ─────────────────────────────────────────────────


class DWConvMlp(nn.Module):
    """Gated MLP with depthwise conv2d for local spatial mixing.
    Only spatial tokens pass through this; class tokens are excluded."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        # x: (B, H*W, C) — spatial tokens only
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        B, N, D = x.shape
        x = x.transpose(1, 2).view(B, D, H, W)
        x = self.dwconv(x)
        x = x.view(B, D, N).transpose(1, 2)
        x = x * self.act(gate)
        return self.fc2(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK-norm, no positional encoding."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

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
    """Transformer block where class tokens attend but skip MLP.
    Only spatial tokens pass through the DWConv MLP."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = DWConvMlp(dim, dim * 4)

    def forward(self, x, H, W, context=None):
        """
        x: (B, H*W, dim) spatial tokens
        context: (B, K, dim) class tokens or None
        Returns: (x, context)
        """
        n_ctx = 0
        if context is not None:
            x_seq = torch.cat([context, x], dim=1)
            n_ctx = context.shape[1]
        else:
            x_seq = x

        # Self-attention over all tokens (class + spatial)
        x_seq = x_seq + self.self_attn(self.norm1(x_seq))

        # Split: class tokens skip MLP
        if n_ctx > 0:
            context = x_seq[:, :n_ctx]
            x = x_seq[:, n_ctx:]
        else:
            x = x_seq

        # MLP with DWConv (spatial tokens only)
        x = x + self.mlp(self.norm2(x), H, W)

        return x, context


# ─── Small UNet (Pixel Predictor) ───────────────────────────────────────────


class UNetBlock(nn.Module):
    """Residual block: 1×1 projection → 3×3 depthwise conv, ×2, with GroupNorm."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Block 1: proj + DW
        self.proj1 = nn.Conv2d(in_ch, out_ch, 1)
        self.dw1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch)
        self.gn1 = nn.GroupNorm(8, out_ch)
        # Block 2: proj + DW
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
    """Convolutional UNet (f16) conditioned by transformer features.
    Global feature vector is concatenated channel-wise at the bottleneck.
    Uses subpixel (pixel_unshuffle/shuffle) for down/upsampling."""

    def __init__(self, in_channels, cond_dim, base_dim=64):
        super().__init__()
        out_channels = in_channels + 1  # x0 + logvar
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]  # 64,128,256,512

        # Inject spatial cond at input resolution
        self.cond_to_input = nn.Conv2d(cond_dim, in_channels, 1)

        # Project global cond for bottleneck injection
        self.cond_to_bottleneck = nn.Sequential(
            nn.Linear(cond_dim, dims[3]),
            nn.SiLU(),
            nn.Linear(dims[3], dims[3])
        )

        # Encoder  (4 stages → f16)
        self.enc0 = nn.Conv2d(in_channels, dims[0], 3, 1, 1)        # H — simple conv
        self.down0 = nn.Conv2d(dims[0] * 4, dims[0], 1)             # pixel_unshuffle → proj
        self.enc1 = UNetBlock(dims[0], dims[1])                      # H/2
        self.down1 = nn.Conv2d(dims[1] * 4, dims[1], 1)
        self.enc2 = UNetBlock(dims[1], dims[2])                      # H/4
        self.down2 = nn.Conv2d(dims[2] * 4, dims[2], 1)
        self.enc3 = UNetBlock(dims[2], dims[3])                      # H/8
        self.down3 = nn.Conv2d(dims[3] * 4, dims[3], 1)

        # Bottleneck @ H/16 (enc output + global cond concat)
        self.mid = UNetBlock(dims[3] * 2, dims[3])

        # Decoder  (4 stages → back to H)
        self.up3 = nn.Conv2d(dims[3], dims[3] * 4, 1)               # proj → pixel_shuffle
        self.dec3 = UNetBlock(dims[3] * 2, dims[3])                  # skip from enc3
        self.up2 = nn.Conv2d(dims[3], dims[2] * 4, 1)
        self.dec2 = UNetBlock(dims[2] * 2, dims[2])                  # skip from enc2
        self.up1 = nn.Conv2d(dims[2], dims[1] * 4, 1)
        self.dec1 = UNetBlock(dims[1] * 2, dims[1])                  # skip from enc1
        self.up0 = nn.Conv2d(dims[1], dims[0] * 4, 1)
        self.dec0 = nn.Conv2d(dims[0] * 2, dims[0], 3, 1, 1)        # H — simple conv

        self.out_conv = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x, cond_spatial):
        """
        x: (B, in_channels, H, W) noisy input
        cond_spatial: (B, cond_dim, H, W) transformer features
        Returns: (B, in_channels + 1, H, W)
        """
        # Global conditioning vector
        cond_global = cond_spatial.mean(dim=(2, 3))                  # (B, cond_dim)
        cond_bottleneck = self.cond_to_bottleneck(cond_global)       # (B, 512)

        # Add spatial cond at input
        x = x + self.cond_to_input(cond_spatial)

        # Encoder (subpixel downsampling)
        h0 = self.enc0(x)                                            # H
        h1 = self.enc1(self.down0(F.pixel_unshuffle(h0, 2)))         # H/2
        h2 = self.enc2(self.down1(F.pixel_unshuffle(h1, 2)))         # H/4
        h3 = self.enc3(self.down2(F.pixel_unshuffle(h2, 2)))         # H/8

        # Bottleneck: concat global cond channel-wise @ H/16
        hb = self.down3(F.pixel_unshuffle(h3, 2))                   # H/16
        B, _, Hb, Wb = hb.shape
        cond_expand = cond_bottleneck[:, :, None, None].expand(B, -1, Hb, Wb)
        h = self.mid(torch.cat([hb, cond_expand], dim=1))

        # Decoder (subpixel upsampling)
        h = self.dec3(torch.cat([F.pixel_shuffle(self.up3(h), 2), h3], dim=1))   # H/8
        h = self.dec2(torch.cat([F.pixel_shuffle(self.up2(h), 2), h2], dim=1))   # H/4
        h = self.dec1(torch.cat([F.pixel_shuffle(self.up1(h), 2), h1], dim=1))   # H/2
        h = self.dec0(torch.cat([F.pixel_shuffle(self.up0(h), 2), h0], dim=1))   # H

        return self.out_conv(h)


# ─── Main Model ─────────────────────────────────────────────────────────────


class TokenformerDiT(nn.Module):
    """
    3-stage Transformer (Encoder → Mid → Decoder) with PixelShuffle 2x
    between stages, feeding a small UNet that predicts clean x0.

    - patch_size controls spatial patchification (pixel_unshuffle) before
      the transformer; the UNet still operates at full resolution
    - No positional encoding
    - Class tokens attend in self-attention but skip MLP
    - DWConv2D in MLP for spatial locality
    - Timestep conditioning via additive embedding
    """

    def __init__(self, in_channels=128, dim=768, depth=12, num_heads=12,
                 num_classes=12476, use_checkpoint=False,
                 encoder_depth=2, decoder_depth=2,
                 unet_base_dim=64, patch_size=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size

        # Input projection (channels ×p² after pixel_unshuffle)
        self.patch_embed = nn.Linear(in_channels * patch_size ** 2, dim)

        # Conditioning
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = MeanPoolingEmbedder(num_classes, dim)

        # Stage 1: Encoder (at Hp×Wp, dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(encoder_depth)
        ])

        # PixelShuffle Down: (B, Hp*Wp, dim) → (B, Hp/2*Wp/2, dim)
        self.down_proj = nn.Linear(dim * 4, dim)

        # Stage 2: Mid (at Hp/2×Wp/2, dim)
        mid_depth = depth - encoder_depth - decoder_depth
        self.mid_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(mid_depth)
        ])

        # PixelShuffle Up: (B, Hp/2*Wp/2, dim) → (B, Hp*Wp, dim)
        self.up_proj = nn.Linear(dim, dim * 4)

        # Skip fusion: concat(skip, upsampled) → Linear → dim
        self.skip_fusion = nn.Linear(dim * 2, dim)

        # Stage 3: Decoder (at Hp×Wp, dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(decoder_depth)
        ])

        # Final norm → project back to spatial cond channels, then pixel_shuffle
        self.final_norm = RMSNorm(dim)
        self.cond_proj = nn.Linear(dim, dim * patch_size ** 2)

        # Small UNet pixel predictor (full resolution)
        self.unet = SmallUNet(in_channels, dim, unet_base_dim)

    def _run_block(self, block, x, H, W, context):
        if self.use_checkpoint and self.training:
            return checkpoint(block, x, H, W, context, use_reentrant=False)
        return block(x, H, W, context)

    def forward(self, x_in, t, y_indices, y_offsets=None):
        """
        Args:
            x_in: (B, C, H, W) noisy input (full resolution)
            t: (B,) scalar timesteps
            y_indices, y_offsets: class tag indices/offsets
        Returns:
            x0_pred: (B, C, H, W) predicted clean image
            logvar: (B, 1, H, W) log-variance for NLL
        """
        B, C, H, W = x_in.shape
        p = self.patch_size

        # 1. Patchify via pixel_unshuffle → (B, C*p², H/p, W/p)
        if p > 1:
            x_patch = F.pixel_unshuffle(x_in, p)
        else:
            x_patch = x_in
        Hp, Wp = H // p, W // p

        # 2. Flatten + embed + add timestep
        x = x_patch.flatten(2).transpose(1, 2)         # (B, Hp*Wp, C*p²)
        x = self.patch_embed(x)                         # (B, Hp*Wp, dim)
        t_emb = self.t_embedder(t)                      # (B, dim)
        x = x + t_emb.unsqueeze(1)

        # 3. Class tokens (attend only, skip MLP)
        context = self.y_embedder(y_indices, y_offsets)  # (B, 4, dim)

        # 4. Encoder blocks at Hp×Wp
        for block in self.encoder_blocks:
            x, context = self._run_block(block, x, Hp, Wp, context)
        skip = x

        # 5. PixelShuffle Down: Hp×Wp → Hp/2×Wp/2
        x = x.transpose(1, 2).view(B, self.dim, Hp, Wp)
        x = F.pixel_unshuffle(x, 2)
        Hm, Wm = Hp // 2, Wp // 2
        x = x.flatten(2).transpose(1, 2)
        x = self.down_proj(x)

        # 6. Mid blocks at Hp/2×Wp/2
        for block in self.mid_blocks:
            x, context = self._run_block(block, x, Hm, Wm, context)

        # 7. PixelShuffle Up: Hp/2×Wp/2 → Hp×Wp
        x = self.up_proj(x)
        x = x.transpose(1, 2).view(B, self.dim * 4, Hm, Wm)
        x = F.pixel_shuffle(x, 2)
        x = x.flatten(2).transpose(1, 2)

        # 8. Skip fusion
        x = self.skip_fusion(torch.cat([x, skip], dim=-1))

        # 9. Decoder blocks at Hp×Wp
        for block in self.decoder_blocks:
            x, context = self._run_block(block, x, Hp, Wp, context)

        # 10. Produce spatial conditioning features at full resolution
        x = self.final_norm(x)
        x = self.cond_proj(x)                           # (B, Hp*Wp, dim*p²)
        x = x.transpose(1, 2).view(B, self.dim * p ** 2, Hp, Wp)
        if p > 1:
            cond_features = F.pixel_shuffle(x, p)       # (B, dim, H, W)
        else:
            cond_features = x

        # 11. Small UNet predicts x0 + logvar (full resolution)
        out = self.unet(x_in, cond_features)
        x0_pred = out[:, :self.in_channels]
        logvar = out[:, self.in_channels:]

        return x0_pred, logvar


# ─── Tag Processor ──────────────────────────────────────────────────────────


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


# ─── Sampling ────────────────────────────────────────────────────────────────


@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=50, guidance_scale=1.5, noise=None, cfg_scale=0,
                sampler_type="euler", **kwargs):
    """Flow matching sampling with direct x0 prediction."""
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

    for i in tqdm(range(steps)):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size,), t_curr.item(), device=device, dtype=x.dtype)

        # Predict x0 (conditioned)
        x0_cond, _ = model(x, t_vec, y_indices, y_offsets)

        # Predict x0 (unconditioned)
        x0_uncond, _ = model(x, t_vec, y_null_indices, y_null_offsets)

        # CFG in x0 space
        x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

        # Convert to velocity: v = (x0 - xt) / (1 - t), then step
        # x_t = (1-t)*x0 + t*noise => v = dx/dt = noise - x0 = (xt - x0) / t ... 
        # Actually for flow: v = x0 - noise, and xt = (1-t)*x0 + t*noise
        # So noise = (xt - (1-t)*x0) / t, and v_flow = noise - x0
        # Step: x_{t+dt} = x_t + dt * v
        # v = (xt - x0) / t  (when t > 0)
        t_safe = t_curr.clamp(min=0.05)
        v = (x - x0_guided) / t_safe
        x = x + dt * v

    model.train()
    return x


# ─── Smoke Test ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Initializing DiT on CPU...")
    device = torch.device("cpu")

    model = TokenformerDiT(
        in_channels=3,
        dim=1024,
        depth=24,
        num_heads=16,
        num_classes=12476,
        encoder_depth=2,
        decoder_depth=2,
        unet_base_dim=64,
    ).to(device)

    num_params = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params / 1e6:.2f} M")

    batch_size = 2
    latent_size = 256

    x_in = torch.randn(batch_size, 3, latent_size, latent_size, device=device)
    t = torch.rand(batch_size, device=device)

    y_indices = torch.randint(0, 12476, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)

    print("\nRunning forward pass (training)...")
    model.train()
    x0_pred, logvar = model(x_in, t, y_indices, y_offsets)

    print(f"Input shape:  {x_in.shape}")
    print(f"Output x0_pred shape: {x0_pred.shape}")
    print(f"Output logvar shape:  {logvar.shape}")

    # Quick backward test
    loss = x0_pred.mean() + logvar.mean()
    loss.backward()
    print("✅ Forward + backward pass completed successfully!")