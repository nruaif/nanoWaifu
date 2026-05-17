import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Binary quantization with temperature annealing
# ==========================================================
torch._functorch.config.activation_memory_budget = 0.5
class DCAEDownsample(nn.Module):
    """
    DC-AE Downsample Block with Residual Autoencoding.
    Assumes a 2x spatial downsample and 2x channel increase (C -> 2C).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Main Neural Network Path
        self.main_path = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, 1)
        )
        
        # Non-parametric shortcut operations
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        main_out = self.main_path(x)
        
        # Shortcut Path [cite: 198, 199, 200]
        # 1. Space-to-Channel: (B, C, H, W) -> (B, 4C, H/2, W/2)
        unshuffled = self.unshuffle(x) 
        
        # 2. Channel Averaging: Split 4C into two groups of 2C and average them
        chunk1, chunk2 = unshuffled.chunk(2, dim=1)
        shortcut_out = (chunk1 + chunk2) / 2.0
        
        return main_out + shortcut_out


class DCAEUpsample(nn.Module):
    """
    DC-AE Upsample Block with Residual Autoencoding.
    Assumes a 2x spatial upsample and 2x channel decrease (2C -> C).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Main Neural Network Path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1),
            nn.PixelShuffle(2)
        )
        
        # Non-parametric shortcut operations
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        main_out = self.main_path(x)
        
        # Shortcut Path [cite: 202, 203, 204]
        # 1. Channel-to-Space: (B, 2C, H/2, W/2) -> (B, C/2, H, W)
        # Note: PixelShuffle(2) expects channels to be divisible by 4.
        shuffled = self.shuffle(x) 
        
        # 2. Channel Duplicating: Duplicate C/2 to C via concatenation
        shortcut_out = torch.cat([shuffled, shuffled], dim=1)
        
        return main_out + shortcut_out
        
class BinarySTE(torch.autograd.Function):
    """Binary Straight-Through Estimator with tanh surrogate gradient.
    Produces {-1, +1} values with smooth backward pass."""
    @staticmethod
    def forward(ctx, x, temp):
        ctx.save_for_backward(x, temp)
        y = torch.where(x >= 0,
                        torch.ones_like(x),
                        -torch.ones_like(x))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, temp = ctx.saved_tensors
        tanh_val = torch.tanh(x / temp)
        surrogate_grad = (1.0 - tanh_val.pow(2)) / temp
        return grad_output * surrogate_grad, None


class AdaptiveBitwiseSign(nn.Module):
    """Binary quantizer with annealing temperature for gradual sharpening."""
    def __init__(self, initial_temp=1.0):
        super().__init__()
        self.register_buffer('temp', torch.tensor(initial_temp, dtype=torch.float32))

    def forward(self, x):
        return BinarySTE.apply(x, self.temp)

    def anneal_temp(self, factor=0.98, min_temp=0.01):
        """Call each epoch to gradually sharpen quantization."""
        self.temp.mul_(factor).clamp_(min=min_temp)


# ==========================================================
# InceptionNeXt Blocks
# ==========================================================

class InceptionDWConv2d(nn.Module):
    """
    Inception depthwise convolution from the InceptionNeXt paper.
    Splits channels into 4 branches: 3x3 square, 1x11 band, 11x1 band, and Identity.
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)

        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size,
            padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc, gc, kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2), groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0), groups=gc
        )

        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)

    def forward(self, x):
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (self.dwconv_hw(x_hw),
             self.dwconv_w(x_w),
             self.dwconv_h(x_h),
             x_id),
            dim=1
        )


class InceptionNeXtBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        self.token_mixer = InceptionDWConv2d(dim)
        self.norm = nn.BatchNorm2d(dim)

        hidden_dim = int(dim * expansion_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x + shortcut


# ==========================================================
# Transformer Block (operates on [B, N, C] token sequences)
# ==========================================================

class TransformerBlock(nn.Module):
    """Transformer block operating on token sequences [B, N, C]."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    @torch.compile
    def forward(self, x):
        # x: [B, N, C]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ==========================================================
# Multi-Scale CNN Encoder (InceptionNeXt stages only, no transformers)
# ==========================================================

class MultiScaleCNNEncoder(nn.Module):
    def __init__(self, patch=4, dims=(96, 192, 384, 768), depths=(3, 3, 9, 3)):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(patch)
        self.stem = nn.Conv2d(3 * patch * patch, dims[0], 7, padding=3)

        self.initial_blocks = nn.Sequential(
            *[InceptionNeXtBlock(dims[0]) for _ in range(depths[0])]
        )

        self.stages = nn.ModuleList()
        for i in range(len(dims) - 1):
            # REPLACED: Use the new DC-AE downsample block
            downsample = DCAEDownsample(dims[i], dims[i + 1])
            
            blocks = nn.Sequential(
                *[InceptionNeXtBlock(dims[i + 1]) for _ in range(depths[i + 1])]
            )
            self.stages.append(nn.ModuleList([downsample, blocks]))
            
    @torch.compile
    def forward(self, x):
        x = self.unshuffle(x)
        x = self.stem(x)
        x = self.initial_blocks(x)

        for downsample, blocks in self.stages:
            x = downsample(x)
            x = blocks(x)

        return x  # [B, dims[-1], H, W]


class MultiScaleCNNDecoder(nn.Module):
    def __init__(self, patch=4, dims=(96, 192, 384, 768), depths=(3, 3, 9, 3)):
        super().__init__()

        self.stages = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            # REPLACED: Use the new DC-AE upsample block
            upsample = DCAEUpsample(dims[i], dims[i - 1])
            
            blocks = nn.Sequential(
                *[InceptionNeXtBlock(dims[i - 1]) for _ in range(depths[i - 1])]
            )
            self.stages.append(nn.ModuleList([upsample, blocks]))

        self.to_pixels = nn.Conv2d(dims[0], 3 * patch * patch, 3, padding=1)
        self.shuffle = nn.PixelShuffle(patch)
        
    @torch.compile
    def forward(self, x):
        # x: [B, dims[-1], H, W]
        for upsample, blocks in self.stages:
            x = upsample(x)
            x = blocks(x)

        img = self.shuffle(self.to_pixels(x))
        return img


# ==========================================================
# Main Autoencoder
# ==========================================================

class BinaryAutoencoder(nn.Module):
    def __init__(
        self,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        latent_discrete=256,
        num_transformer_blocks=8,
        num_cls_tokens=4,
        mask_block_size=14,
        use_masking=True,
    ):
        super().__init__()

        self.num_cls_tokens = num_cls_tokens
        self.mask_block_size = mask_block_size
        self.use_masking = use_masking

        token_dim = dims[-1]

        # --------------------------------------------------
        # CNN backbone (no transformers)
        # --------------------------------------------------
        self.encoder_cnn = MultiScaleCNNEncoder(dims=dims, depths=depths)
        self.decoder_cnn = MultiScaleCNNDecoder(dims=dims, depths=depths)

        # --------------------------------------------------
        # Transformer stacks (operate on [B, N, C] tokens)
        # --------------------------------------------------
        self.encoder_transformers = nn.ModuleList([
            TransformerBlock(token_dim, num_heads=8)
            for _ in range(num_transformer_blocks)
        ])
        self.decoder_transformers = nn.ModuleList([
            TransformerBlock(token_dim, num_heads=8)
            for _ in range(num_transformer_blocks)
        ])

        # --------------------------------------------------
        # CLS tokens and mask token
        # --------------------------------------------------
        self.cls_tokens = nn.Parameter(torch.randn(1, num_cls_tokens, token_dim))

        # --------------------------------------------------
        # Spatial bottleneck (ternary discrete + continuous bypass)
        # --------------------------------------------------
        self.to_latent_discrete = nn.Conv2d(token_dim, latent_discrete, 3, padding=1)
        self.from_latent = nn.Conv2d(latent_discrete, token_dim, 1)
        self.quant = AdaptiveBitwiseSign()

# ======================================================
    # Channel-wise left-to-right block masking on z_discrete
    # ======================================================
    def apply_masking(self, z, B, override_num_blocks=None):
        C = z.shape[1]
        block = self.mask_block_size
        num_blocks = C // block

        if num_blocks == 0:
            return z, None

        z = z.clone()

        # 1. Determine masking behavior
        if override_num_blocks is not None:
            # --- OVERRIDE BEHAVIOR ---
            # Mask ALL samples in the batch
            num_masked = B
            masked_ids = torch.arange(B, device=z.device)
            
            # Clamp the requested blocks to drop to valid bounds [0, max_blocks]
            n_blocks_to_drop = max(0, min(override_num_blocks, num_blocks))
            
            if n_blocks_to_drop == 0:
                return z, None
                
            # Create a uniform array so every sample drops the exact same amount
            n_mask = torch.full((B,), n_blocks_to_drop, dtype=torch.long, device=z.device)

        else:
            # --- DEFAULT TRAINING BEHAVIOR ---
            if (not self.training) or B < 2 or (not self.use_masking):
                return z, None
            
            num_masked = B // 2
            idx = torch.randperm(B, device=z.device)
            masked_ids = idx[:num_masked]

            # ---- Desired peak ----
            target_remaining_channels = 112 // 2

            mean_mask_blocks = (C - target_remaining_channels) / block
            std_mask_blocks = num_blocks * 0.15

            # Gaussian sampling for variable dropping
            n_mask = torch.randint(
            low=1,
            high=num_blocks + 1,
            size=(num_masked,),
            device=z.device,
)

            # round + clamp

        # 2. Apply the mask
        for i in range(num_masked):
            start_ch = C - n_mask[i].item() * block
            z[masked_ids[i], start_ch:] = 0.0

        return z, masked_ids

    # ======================================================
    # forward
    # ======================================================
    def forward(self, x, num_to_drop=None):
        # 1. CNN encode
        features = self.encoder_cnn(x)  # [B, C, H, W]
        B, C, H, W = features.shape

        # 2. Encoder transformers with CLS tokens
        spatial = features.flatten(2).transpose(1, 2)  # [B, HW, C]
        cls = self.cls_tokens.expand(B, -1, -1)         # [B, num_cls, C]
        tokens = torch.cat([cls, spatial], dim=1)       # [B, num_cls + HW, C]

        for block in self.encoder_transformers:
            tokens = block(tokens)

        # Separate CLS and spatial
        enc_cls = tokens[:, :self.num_cls_tokens]
        spatial = tokens[:, self.num_cls_tokens:]
        features = spatial.transpose(1, 2).reshape(B, C, H, W)

        # 3. Spatial bottleneck (discrete only) + channel-block masking
        z_discrete = self.quant(self.to_latent_discrete(features))
        
        # Pass the override parameter here
        z_discrete, masked_ids = self.apply_masking(z_discrete, B, override_num_blocks=num_to_drop)
        
        features = self.from_latent(z_discrete)  # [B, C, H, W]

        # 4. Decoder transformers with CLS tokens (no token-level masking)
        spatial = features.flatten(2).transpose(1, 2)
        tokens = torch.cat([enc_cls, spatial], dim=1)

        # (masking already applied at the latent level above)

        for block in self.decoder_transformers:
            tokens = block(tokens)

        # Remove CLS tokens
        spatial = tokens[:, self.num_cls_tokens:]
        features = spatial.transpose(1, 2).reshape(B, C, H, W)

        # 5. CNN decode
        img = self.decoder_cnn(features)
        img = torch.clamp(img, -1, 1)

        return img, z_discrete, masked_ids


# ==========================================================
# PatchGAN Discriminator
# ==========================================================

class PatchDiscriminator(nn.Module):
    """Multi-scale PatchGAN discriminator with spectral normalization.

    Tricks used:
    - Spectral normalization on all conv layers for Lipschitz constraint
    - No BatchNorm (spectral norm replaces it for stability)
    - LeakyReLU(0.2) following DCGAN/pix2pix convention
    - Multi-scale: runs at original + downsampled resolution
    - Outputs spatial grid of logits (not single scalar)
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3, num_scales=2):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for _ in range(num_scales):
            self.discriminators.append(self._make_disc(in_channels, ndf, n_layers))

        # Learnable downsampler for multi-scale
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def _make_disc(self, in_channels, ndf, n_layers):
        layers = []

        # First layer (no normalization on input layer per DCGAN convention)
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers with increasing channels
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 2 if i < n_layers - 1 else 1

            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=stride, padding=1)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final 1-channel prediction layer
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)
        ))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Returns list of patch logit maps, one per scale."""
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.num_scales - 1:
                x = self.downsample(x)
        return outputs


def disc_hinge_loss(real_preds, fake_preds):
    """Hinge loss for discriminator. Pushes real > 1, fake < -1."""
    loss = 0
    for rp, fp in zip(real_preds, fake_preds):
        loss += F.relu(1.0 - rp).mean() + F.relu(1.0 + fp).mean()
    return loss / len(real_preds)


def gen_hinge_loss(fake_preds):
    """Hinge loss for generator. Pushes fake predictions higher."""
    loss = 0
    for fp in fake_preds:
        loss += -fp.mean()
    return loss / len(fake_preds)


def r1_gradient_penalty(real_images, real_preds):
    """R1 gradient penalty for regularization (Mescheder et al. 2018).
    Penalizes the gradient of D on real images to prevent mode collapse."""
    total_pred = sum(rp.sum() for rp in real_preds)

    grads = torch.autograd.grad(
        outputs=total_pred,
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    penalty = grads.pow(2).reshape(grads.shape[0], -1).sum(1).mean()
    return penalty


# ==========================================================
# test
# ==========================================================
if __name__ == "__main__":
    model = BinaryAutoencoder().cuda()
    disc = PatchDiscriminator().cuda()
    model.train()

    x = torch.randn(4, 3, 256, 256).cuda()
    y, latent, masked_ids = model(x)

    print("input :", x.shape)
    print("recon :", y.shape)
    print("latent:", latent.shape)
    print(f"masked_ids: {masked_ids}")

    preds = disc(y)
    print(f"disc scales: {len(preds)}, shapes: {[p.shape for p in preds]}")

    total_ae = sum(p.numel() for p in model.parameters())
    total_disc = sum(p.numel() for p in disc.parameters())
    print(f"AE params: {total_ae / 1e6:.1f}M, Disc params: {total_disc / 1e6:.1f}M")