import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Ternary quantization with temperature annealing
# ==========================================================

class TernarySTE(torch.autograd.Function):
    """Ternary Straight-Through Estimator with sigmoid surrogate gradient.
    Produces {-1, 0, +1} values with smooth backward pass."""
    @staticmethod
    def forward(ctx, x, temp):
        ctx.save_for_backward(x, temp)
        return torch.sign(x) * (x.abs() > 1).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, temp = ctx.saved_tensors

        def sigmoid_derivative(z):
            s = torch.sigmoid(z)
            return s * (1.0 - s)

        surrogate_grad = (sigmoid_derivative((x - 1.0) / temp) +
                          sigmoid_derivative((x + 1.0) / temp)) / temp

        return grad_output * surrogate_grad, None


class AdaptiveBitwiseSign(nn.Module):
    """Ternary quantizer with annealing temperature for gradual sharpening."""
    def __init__(self, initial_temp=1.0):
        super().__init__()
        self.register_buffer('temp', torch.tensor(initial_temp, dtype=torch.float32))

    def forward(self, x):
        return TernarySTE.apply(x, self.temp)

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
# Transformer Block for Final Stage
# ==========================================================

class TransformerBlock(nn.Module):
    """Transformer block operating on spatial feature maps."""
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

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


# ==========================================================
# Multi-Scale Encoder & Decoder (with Sub-Pixel Sampling)
# ==========================================================

class MultiScaleEncoder(nn.Module):
    def __init__(self, patch=4, dims=(96, 192, 384, 768), depths=(3, 3, 9, 3),
                 latent_discrete=256, latent_continuous=32, num_transformer_blocks=8):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(patch)
        self.stem = nn.Conv2d(3 * patch * patch, dims[0], 7, padding=3)

        self.initial_blocks = nn.Sequential(
            *[InceptionNeXtBlock(dims[0]) for _ in range(depths[0])]
        )

        self.stages = nn.ModuleList()
        for i in range(len(dims) - 1):
            downsample = nn.Sequential(
                nn.PixelUnshuffle(2),
                nn.Conv2d(dims[i] * 4, dims[i + 1], 1)
            )
            if i == len(dims) - 2:  # Last stage: add transformer blocks
                blocks = nn.Sequential(
                    *[InceptionNeXtBlock(dims[i + 1]) for _ in range(depths[i + 1])],
                    *[TransformerBlock(dims[i + 1], num_heads=8) for _ in range(num_transformer_blocks)]
                )
            else:
                blocks = nn.Sequential(
                    *[InceptionNeXtBlock(dims[i + 1]) for _ in range(depths[i + 1])]
                )
            self.stages.append(nn.ModuleList([downsample, blocks]))

        self.to_latent_discrete = nn.Conv2d(dims[-1], latent_discrete, 3, padding=1)
        self.to_latent_continuous = nn.Conv2d(dims[-1], latent_continuous, 3, padding=1)

        self.quant = AdaptiveBitwiseSign()

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.stem(x)
        x = self.initial_blocks(x)

        for downsample, blocks in self.stages:
            x = downsample(x)
            x = blocks(x)

        z_discrete = self.quant(self.to_latent_discrete(x))
        z_continuous = self.to_latent_continuous(x)

        return z_discrete, z_continuous


class MultiScaleDecoder(nn.Module):
    def __init__(self, patch=4, dims=(96, 192, 384, 768), depths=(3, 3, 9, 3),
                 latent_discrete=256, latent_continuous=32, num_transformer_blocks=8):
        super().__init__()
        total_latent = latent_discrete + latent_continuous
        self.from_latent = nn.Conv2d(total_latent, dims[-1], 1)

        self.initial_blocks = nn.Sequential(
            *[InceptionNeXtBlock(dims[-1]) for _ in range(depths[-1])],
            *[TransformerBlock(dims[-1], num_heads=8) for _ in range(num_transformer_blocks)]
        )

        self.stages = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            upsample = nn.Sequential(
                nn.Conv2d(dims[i], dims[i - 1] * 4, 1),
                nn.PixelShuffle(2)
            )
            blocks = nn.Sequential(
                *[InceptionNeXtBlock(dims[i - 1]) for _ in range(depths[i - 1])]
            )
            self.stages.append(nn.ModuleList([upsample, blocks]))

        self.to_pixels = nn.Conv2d(dims[0], 3 * patch * patch, 3, padding=1)
        self.shuffle = nn.PixelShuffle(patch)

    def forward(self, z_discrete, z_continuous):
        z = torch.cat([z_discrete, z_continuous], dim=1)
        x = self.from_latent(z)
        x = self.initial_blocks(x)

        for upsample, blocks in self.stages:
            x = upsample(x)
            x = blocks(x)

        img = self.shuffle(self.to_pixels(x))
        return img


# ==========================================================
# Main Autoencoder
# ==========================================================

class BinaryAutoencoder(nn.Module):
    def __init__(self, dims=(96, 192, 384, 768), depths=(3, 3, 9, 3),
                 latent_discrete=256, latent_continuous=32,
                 residual_dropout_prob=0.1, num_transformer_blocks=8):
        super().__init__()
        self.encoder = MultiScaleEncoder(
            dims=dims, depths=depths,
            latent_discrete=latent_discrete,
            latent_continuous=latent_continuous,
            num_transformer_blocks=num_transformer_blocks,
        )
        self.decoder = MultiScaleDecoder(
            dims=dims, depths=depths,
            latent_discrete=latent_discrete,
            latent_continuous=latent_continuous,
            num_transformer_blocks=num_transformer_blocks,
        )
        self.latent_discrete = latent_discrete
        self.latent_continuous = latent_continuous
        self.residual_dropout_prob = residual_dropout_prob

    def forward(self, x):
        z_discrete, z_continuous = self.encoder(x)

        # Residual dropout: randomly zero continuous channels per-sample
        if self.training and self.residual_dropout_prob > 0:
            batch_size = z_continuous.shape[0]
            mask = torch.rand(batch_size, 1, 1, 1, device=z_continuous.device) > self.residual_dropout_prob
            z_continuous = z_continuous * mask.float()

        recon = self.decoder(z_discrete, z_continuous)
        recon = torch.clamp(recon, -1, 1)

        return recon, z_discrete

    def anneal_temperature(self, factor=0.98, min_temp=0.01):
        self.encoder.quant.anneal_temp(factor=factor, min_temp=min_temp)


# ==========================================================
# test
# ==========================================================
if __name__ == "__main__":
    model = BinaryAutoencoder().cuda()
    model.train()

    x = torch.randn(4, 3, 256, 256).cuda()

    y, latent = model(x)

    print("input :", x.shape)
    print("recon :", y.shape)
    print("latent:", latent.shape)
    print(f"latent nonzero ratio: {(latent != 0).float().mean().item():.3f}")
    print(f"latent unique values: {latent.unique().tolist()}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.1f}M")