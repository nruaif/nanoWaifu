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
class LayerNorm2d(nn.LayerNorm):
    """Channels-last LayerNorm applied in NCHW layout."""
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def modulate(x, shift, scale):
    """Apply adaptive modulation: x * (1 + scale) + shift with spatial broadcast."""
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


# =============================================================================
# Embedding Layers for Timesteps and Tags
# =============================================================================
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal embedding + MLP."""
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


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
# Core FCDM Components
# =============================================================================
class GRN(nn.Module):
    """GRN (Global Response Normalization) layer in NCHW format."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block with adaLN-Zero conditioning (all NCHW).
    DWConv → Norm+Modulate → Expand(1x1) → GELU → GRN → Reduce(1x1) → gate → residual.
    """
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, affine=False, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.act = nn.GELU()
        self.grn = GRN(int(dim * mlp_ratio))
        self.pwconv2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, C, H, W) feature map
        c: (B, C) conditioning vector
        """
        h = self.dwconv(x)
        # adaLN-Zero: compute shift, scale, gate from conditioning
        shift, scale, gate = self.adaLN_modulation(c).unsqueeze(2).unsqueeze(3).chunk(3, dim=1)
        h = self.norm(h)
        h = torch.addcmul(shift, h, scale + 1)
        # Pointwise MLP
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        # Gate and residual
        h = h * gate
        return x + h


class ConvFinalLayer(nn.Module):
    """Conv-style final layer with adaLN modulation (shift + scale only, no gate)."""
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Spatial downsample via Conv2d + PixelUnshuffle (2x)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsample via Conv2d + PixelShuffle (2x)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.conv(x)


# =============================================================================
# FCDM U-Net Backbone
# =============================================================================
class FCDM(nn.Module):
    """
    Fully Convolutional Diffusion Model (FCDM).
    ConvNeXt-based U-Net with adaLN-Zero conditioning, PixelShuffle up/down,
    concat skip connections, and per-resolution conditioning.
    """

    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 128,
        depth=[2, 4, 8, 4, 2],
        num_classes: int = 1000,
        use_checkpoint: bool = False,
        max_tags: int = 64,
        mlp_ratio: float = 3.0,
        learn_sigma: bool = False,
        **kwargs,  # absorb old params: expansion_ratio, kernel_size, cond_dim, fcdm_blocks
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.dim = dim
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint

        # Normalize depth: int → [d, 2d, 4d, 2d, d], list of 5 used directly
        if isinstance(depth, int):
            depth = [depth, depth * 2, depth * 4, depth * 2, depth]
        assert len(depth) == 5, f"depth must be int or list of 5, got {depth}"

        # ---- Input projection ------------------------------------------------
        self.x_embedder = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)

        # ---- Per-resolution timestep embedders --------------------------------
        self.t_embedder_1 = TimestepEmbedder(dim)
        self.t_embedder_2 = TimestepEmbedder(dim * 2)
        self.t_embedder_3 = TimestepEmbedder(dim * 4)

        # ---- Tag conditioning: shared EmbeddingBag + per-resolution projections -
        self.y_embedder = FCDMConditioning(
            num_classes=num_classes, dim=dim, max_tags=max_tags,
        )
        self.y_proj_2 = nn.Linear(dim, dim * 2)
        self.y_proj_3 = nn.Linear(dim, dim * 4)

        # ---- Encoder level 1 -------------------------------------------------
        self.encoder_level_1 = nn.ModuleList([
            ConvNeXtBlock(dim, mlp_ratio=mlp_ratio) for _ in range(depth[0])
        ])
        self.down1_2 = Downsample(dim, dim * 2)

        # ---- Encoder level 2 -------------------------------------------------
        self.encoder_level_2 = nn.ModuleList([
            ConvNeXtBlock(dim * 2, mlp_ratio=mlp_ratio) for _ in range(depth[1])
        ])
        self.down2_3 = Downsample(dim * 2, dim * 4)

        # ---- Bottleneck (latent) ---------------------------------------------
        self.latent = nn.ModuleList([
            ConvNeXtBlock(dim * 4, mlp_ratio=mlp_ratio) for _ in range(depth[2])
        ])

        # ---- Decoder level 2 -------------------------------------------------
        self.up3_2 = Upsample(dim * 4, dim * 2)
        self.reduce_chans_2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.decoder_level_2 = nn.ModuleList([
            ConvNeXtBlock(dim * 2, mlp_ratio=mlp_ratio) for _ in range(depth[3])
        ])

        # ---- Decoder level 1 -------------------------------------------------
        self.up2_1 = Upsample(dim * 2, dim)
        self.reduce_chans_1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder_level_1 = nn.ModuleList([
            ConvNeXtBlock(dim, mlp_ratio=mlp_ratio) for _ in range(depth[4])
        ])

        # ---- Output -----------------------------------------------------------
        self.output_layer = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.final_layer = ConvFinalLayer(dim, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Xavier init for all Linear and Conv2d
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Input projection (re-init as flat matrix)
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Tag embedding (re-init after _basic_init overwrote it)
        nn.init.normal_(self.y_embedder.embedding_bag.weight, std=0.02)
        with torch.no_grad():
            self.y_embedder.embedding_bag.weight[self.y_embedder.padding_idx].zero_()

        # Timestep embedding MLPs
        for t_emb in [self.t_embedder_1, self.t_embedder_2, self.t_embedder_3]:
            nn.init.normal_(t_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(t_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in all blocks
        all_blocks = (
            list(self.encoder_level_1) + list(self.encoder_level_2) +
            list(self.latent) +
            list(self.decoder_level_2) + list(self.decoder_level_1)
        )
        for block in all_blocks:
            if hasattr(block, 'adaLN_modulation'):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

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
        """
        Forward pass of FCDM.
        x_in: (B, C, H, W) input (images or latent representations)
        t: (B,) diffusion timesteps in [0, 1]
        y_indices: flat 1-D tag indices
        y_offsets: per-sample offsets into y_indices
        """
        # Project input
        x = self.x_embedder(x_in)

        # Per-resolution conditioning: time + tags
        t_scaled = t * 1000.0
        y_base = self.y_embedder(y_indices, y_offsets)  # [B, dim]

        c1 = self.t_embedder_1(t_scaled) + y_base
        c2 = self.t_embedder_2(t_scaled) + self.y_proj_2(y_base)
        c3 = self.t_embedder_3(t_scaled) + self.y_proj_3(y_base)

        # Encoder level 1
        out_enc_level1 = x
        for block in self.encoder_level_1:
            out_enc_level1 = self._block_forward(block, out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # Encoder level 2
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            out_enc_level2 = self._block_forward(block, out_enc_level2, c2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # Bottleneck
        latent = inp_enc_level3
        for block in self.latent:
            latent = self._block_forward(block, latent, c3)

        # Decoder level 2 (concat skip)
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = inp_dec_level2[:, :, :out_enc_level2.shape[2], :out_enc_level2.shape[3]]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        out_dec_level2 = self.reduce_chans_2(inp_dec_level2)
        for block in self.decoder_level_2:
            out_dec_level2 = self._block_forward(block, out_dec_level2, c2)

        # Decoder level 1 (concat skip)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = inp_dec_level1[:, :, :out_enc_level1.shape[2], :out_enc_level1.shape[3]]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.reduce_chans_1(inp_dec_level1)
        for block in self.decoder_level_1:
            out_dec_level1 = self._block_forward(block, out_dec_level1, c1)

        # Output
        x = self.output_layer(out_dec_level1)
        x0_pred = self.final_layer(x, c1)

        # Compatibility returns
        infonce_loss = torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)

        if return_features:
            return x0_pred, x
        if return_layer_match:
            return x0_pred, infonce_loss
        return x0_pred


# Backward-compat alias so old imports don't break
TokenformerDiT = FCDM


# =============================================================================
# FCDM Model Configs
# =============================================================================
def FCDM_S(**kwargs):
    return FCDM(dim=128, depth=[2, 4, 8, 4, 2], **kwargs)

def FCDM_B(**kwargs):
    return FCDM(dim=256, depth=[2, 4, 8, 4, 2], **kwargs)

def FCDM_L(**kwargs):
    return FCDM(dim=512, depth=[2, 4, 8, 4, 2], **kwargs)

def FCDM_XL(**kwargs):
    return FCDM(dim=512, depth=[3, 6, 12, 6, 3], **kwargs)

FCDM_models = {
    'FCDM-S': FCDM_S,
    'FCDM-B': FCDM_B,
    'FCDM-L': FCDM_L,
    'FCDM-XL': FCDM_XL,
}


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
        depth=[2, 4, 8, 4, 2],
        num_classes=1000,
        max_tags=64,
        mlp_ratio=3.0,
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

    # Test 4: learn_sigma
    model_sigma = FCDM(in_channels=4, dim=64, depth=[1, 2, 4, 2, 1], num_classes=100, learn_sigma=True).to(device)
    out = model_sigma(x_in, t, y_indices, y_offsets)
    assert out.shape == (batch_size, 8, H, W), f"Expected 8 channels with learn_sigma, got {out.shape}"
    print(f"  [OK] learn_sigma: {out.shape}")

    # Test 5: depth as int (backward compat)
    model_int = FCDM(in_channels=4, dim=64, depth=2, num_classes=100).to(device)
    out = model_int(x_in, t, y_indices, y_offsets)
    assert out.shape == x_in.shape
    print(f"  [OK] depth=int: {out.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)