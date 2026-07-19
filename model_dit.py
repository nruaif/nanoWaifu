import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
import random

torch._dynamo.config.recompile_limit = 128


# =============================================================================
# Utilities
# =============================================================================
def get_timestep_embedding(t, dim):
    """Sinusoidal timestep embedding."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# =============================================================================
# FCDM Components (ConvNeXt-based)
# =============================================================================
class GlobalResponseNorm(nn.Module):
    """Global Response Normalization (GRN) from ConvNeXt V2."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x: [B, H, W, C]
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
        return x


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for FCDM."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim, bias=True)
        )
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, x, c):
        # x: [B, H, W, C], c: [B, cond_dim]
        x_norm = self.norm(x)
        modulation = self.mlp(c)
        gamma, beta, alpha = modulation.chunk(3, dim=-1)
        gamma = gamma.unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1)
        alpha = alpha.unsqueeze(1).unsqueeze(1)
        x_out = x_norm * (1 + gamma) + beta
        return x_out, alpha


class FCDMBlock(nn.Module):
    """FCDM Block: ConvNeXt with AdaLN conditioning."""
    def __init__(self, dim, cond_dim, expansion_ratio=3, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.adaln = AdaLN(dim, cond_dim)
        self.pwconv_expand = nn.Linear(dim, dim * expansion_ratio)
        self.grn = GlobalResponseNorm(dim * expansion_ratio)
        self.act = nn.GELU()
        self.pwconv_reduce = nn.Linear(dim * expansion_ratio, dim)
        nn.init.zeros_(self.pwconv_reduce.weight)
        nn.init.zeros_(self.pwconv_reduce.bias)

    def forward(self, x, c):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # [B, H, W, C]
        x, alpha = self.adaln(x, c)
        x = self.pwconv_expand(x)
        x = self.grn(x)
        x = self.act(x)
        x = self.pwconv_reduce(x)
        identity = identity.permute(0, 2, 3, 1)
        x = identity + alpha * x
        x = x.permute(0, 3, 1, 2)          # [B, C, H, W]
        return x


class FCDMConditioning(nn.Module):
    """Tag conditioning: EmbeddingBag (mean pooling) followed by an MLP."""
    def __init__(self, num_classes: int, dim: int, max_tags: int = 64, **kwargs):
        super().__init__()
        self.dim = dim
        self.max_tags = max_tags
        self.num_classes = num_classes
        self.padding_idx = num_classes

        self.embedding_bag = nn.EmbeddingBag(
            num_classes + 1, dim, mode='mean', padding_idx=self.padding_idx,
        )
        nn.init.normal_(self.embedding_bag.weight, std=0.02)
        with torch.no_grad():
            self.embedding_bag.weight[self.padding_idx].zero_()

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor = None) -> torch.Tensor:
        if y_offsets is None:
            # Single-sample inference
            indices = y_indices[:self.max_tags]
            offsets = torch.zeros(1, dtype=torch.long, device=y_indices.device)
        else:
            # Clamp per-sample lengths to max_tags and rebuild flat indices
            B = len(y_offsets)
            total_len = len(y_indices)
            next_off = torch.cat([y_offsets[1:], total_len.unsqueeze(0) if isinstance(total_len, torch.Tensor) else torch.tensor([total_len], device=y_offsets.device)])
            lengths = (next_off - y_offsets).clamp(max=self.max_tags)

            chunks = []
            new_offsets = []
            running = 0
            for i in range(B):
                clen = lengths[i].item()
                if clen > 0:
                    chunks.append(y_indices[y_offsets[i]:y_offsets[i] + clen])
                new_offsets.append(running)
                running += clen

            indices = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.long, device=y_indices.device)
            offsets = torch.tensor(new_offsets, dtype=torch.long, device=y_indices.device)

        pooled = self.embedding_bag(indices, offsets)  # [B, dim]
        return self.mlp(pooled)


# =============================================================================
# FCDM U-Net Backbone
# =============================================================================
class FCDM(nn.Module):
    """
    Fully Convolutional Diffusion Model (FCDM).
    ConvNeXt-based U-Net with AdaLN conditioning and additive skip connections.
    Scales via two hyperparameters: depth=L (blocks per stage-1) and dim=C (base channels).
    """

    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 128,
        depth: int = 2,
        num_classes: int = 1000,
        use_checkpoint: bool = False,
        max_tags: int = 64,
        fcdm_blocks: int = 2,
        expansion_ratio: int = 3,
        kernel_size: int = 7,
        cond_dim: int = None,
        **kwargs,  # absorb num_heads, num_cls_tokens etc. from old DiT configs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        cond_dim = cond_dim or dim

        # ---- Input projection ------------------------------------------------
        self.input_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        # ---- Conditioning ----------------------------------------------------
        self.y_embedder = FCDMConditioning(
            num_classes=num_classes,
            dim=cond_dim,
            max_tags=max_tags,
        )

        # ---- Encoder ---------------------------------------------------------
        self.enc1 = nn.ModuleList([
            FCDMBlock(dim, cond_dim, expansion_ratio, kernel_size) for _ in range(depth)
        ])
        self.down1 = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
        )

        self.enc2 = nn.ModuleList([
            FCDMBlock(dim * 2, cond_dim, expansion_ratio, kernel_size) for _ in range(depth * 2)
        ])
        self.down2 = nn.Sequential(
            nn.GroupNorm(1, dim * 2),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
        )

        # ---- Bottleneck ------------------------------------------------------
        self.bottleneck = nn.ModuleList([
            FCDMBlock(dim * 4, cond_dim, expansion_ratio, kernel_size) for _ in range(depth * 4)
        ])

        # ---- Decoder ---------------------------------------------------------
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim * 4, dim * 2, kernel_size=3, padding=1),
        )
        self.dec2 = nn.ModuleList([
            FCDMBlock(dim * 2, cond_dim, expansion_ratio, kernel_size) for _ in range(depth * 2)
        ])

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
        )
        self.dec1 = nn.ModuleList([
            FCDMBlock(dim, cond_dim, expansion_ratio, kernel_size) for _ in range(depth)
        ])

        # ---- Output ----------------------------------------------------------
        self.final_norm = nn.GroupNorm(1, dim)
        self.final_conv = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def _block_forward(self, block, x, c):
        if self.use_checkpoint and self.training:
            return checkpoint(block, x, c, use_reentrant=False)
        return block(x, c)

    def forward(
        self,
        x_in,
        t,
        y_indices,
        y_offsets=None,
        return_features=False,
        return_layer_match=False,
        **kwargs,
    ):
        B, C, H, W = x_in.shape

        # Project input
        x = self.input_conv(x_in)

        # Conditioning: time + tags
        t_emb = get_timestep_embedding(t * 1000.0, self.y_embedder.dim).to(x_in.dtype)
        y_emb = self.y_embedder(y_indices, y_offsets)
        c = t_emb + y_emb  # [B, cond_dim]

        # Encoder
        for block in self.enc1:
            x = self._block_forward(block, x, c)
        skip1 = x
        x = self.down1(x)

        for block in self.enc2:
            x = self._block_forward(block, x, c)
        skip2 = x
        x = self.down2(x)

        # Bottleneck
        for block in self.bottleneck:
            x = self._block_forward(block, x, c)

        # Decoder with additive skips
        x = self.up2(x)
        x = x[:, :, :skip2.shape[2], :skip2.shape[3]]
        x = x + skip2
        for block in self.dec2:
            x = self._block_forward(block, x, c)

        x = self.up1(x)
        x = x[:, :, :skip1.shape[2], :skip1.shape[3]]
        x = x + skip1
        for block in self.dec1:
            x = self._block_forward(block, x, c)

        # Output
        x = self.final_norm(x)
        x0_pred = self.final_conv(x)

        # Layer-match placeholder (kept for training-script compatibility)
        infonce_loss = torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)

        if return_features:
            return x0_pred, x
        if return_layer_match:
            return x0_pred, infonce_loss
        return x0_pred


# Backward-compat alias so old imports don't break
TokenformerDiT = FCDM


# =============================================================================
# Tag Processor
# =============================================================================
class TagProcessor:
    def __init__(self, tags_file):
        with open(tags_file, "r", encoding="utf-8") as f:
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


# =============================================================================
# Sampling (Rectified-Flow / FM sampler)
# =============================================================================
@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=250, guidance_scale=1.5, noise=None):
    in_channels = model.in_channels if hasattr(model, "in_channels") else (
        model.module.in_channels if hasattr(model, "module") else 32)
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
    for i in tqdm(range(steps), desc="Sampling", disable=device.type == "cpu"):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size,), t_curr.item(), device=device, dtype=x.dtype)

        x0_cond = model(x, t_vec, y_indices, y_offsets)
        x0_uncond = model(x, t_vec, y_null_indices, y_null_offsets)

        x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        t_reshaped = t_vec.view(-1, 1, 1, 1).clamp(min=1e-5)
        v = (x - x0) / t_reshaped
        x = x + dt * v

    return x


# =============================================================================
# Quick self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FCDM Test Suite")
    print("=" * 60)
    device = torch.device("cpu")

    model = FCDM(
        in_channels=4,
        dim=128,
        depth=2,
        num_classes=1000,
        max_tags=64,
        fcdm_blocks=2,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    y_params = sum(p.numel() for p in model.y_embedder.parameters() if p.requires_grad)
    print(f"  Total params: {num_params / 1e6:.2f}M | Conditioning: {y_params / 1e6:.3f}M")

    batch_size, H, W = 2, 32, 32
    x_in = torch.randn(batch_size, 4, H, W, device=device)
    y_indices = torch.randint(0, 100, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)
    t = torch.rand(batch_size, device=device)

    # Test 1: basic forward
    model.eval()
    x0_pred = model(x_in, t, y_indices, y_offsets)
    assert x0_pred.shape == x_in.shape
    print(f"  [OK] Forward: {x0_pred.shape}")

    # Test 2: return_layer_match
    x0_pred, infonce = model(x_in, t, y_indices, y_offsets, return_layer_match=True)
    assert x0_pred.shape == x_in.shape
    assert isinstance(infonce, torch.Tensor)
    print(f"  [OK] Layer match placeholder: infonce_loss={infonce.item():.4f}")

    # Test 3: backward
    model.train()
    x0_pred, infonce = model(x_in, t, y_indices, y_offsets, return_layer_match=True)
    loss = F.mse_loss(x0_pred, torch.randn_like(x0_pred)) + 0.2 * infonce
    loss.backward()
    print(f"  [OK] Backward: loss={loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)