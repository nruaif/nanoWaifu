import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ==========================================================
# Model Architecture (from user's code)
# ==========================================================

class TernarySTE(torch.autograd.Function):
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
        grad_x = grad_output * surrogate_grad

        return grad_x, None


class AdaptiveBitwiseSign(nn.Module):
    def __init__(self, initial_temp=1.0):
        super().__init__()
        self.register_buffer('temp', torch.tensor(initial_temp, dtype=torch.float32))

    def forward(self, x):
        return TernarySTE.apply(x, self.temp)

    def anneal_temp(self, factor=0.98):
        self.temp.mul_(factor).clamp_(min=0.99)


class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc, padding_mode='reflect')
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc, padding_mode='reflect')
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc, padding_mode='reflect')
        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)

    def forward(self, x):
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h), x_id), dim=1)


class InceptionNeXtBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        self.token_mixer = InceptionDWConv2d(dim)
        self.norm = nn.BatchNorm2d(dim)
        hidden_dim = int(dim * expansion_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hidden_dim, 1), nn.GELU(), nn.Conv2d(hidden_dim, dim, 1))

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x + shortcut


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self, patch=4, dims=[128, 256, 512, 1024], depths=[4, 4, 4, 4], latent_discrete=128, latent_continuous=64):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(patch)
        self.stem = nn.Conv2d(3 * patch * patch, dims[0], 7, padding=3, padding_mode='reflect')
        self.initial_blocks = nn.Sequential(*[InceptionNeXtBlock(dims[0]) for _ in range(depths[0])])
        self.stages = nn.ModuleList()
        for i in range(len(dims)-1):
            downsample = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(dims[i] * 4, dims[i+1], 1))
            if i == len(dims) - 2:
                blocks = nn.Sequential(*[InceptionNeXtBlock(dims[i+1]) for _ in range(depths[i+1])], *[TransformerBlock(dims[i+1], num_heads=16) for _ in range(8)])
            else:
                blocks = nn.Sequential(*[InceptionNeXtBlock(dims[i+1]) for _ in range(depths[i+1])])
            self.stages.append(nn.ModuleList([downsample, blocks]))
        self.to_latent_discrete = nn.Conv2d(dims[-1], latent_discrete, 3, padding=1, padding_mode='reflect')
        if latent_continuous > 0:
            self.to_latent_continuous = nn.Conv2d(dims[-1], latent_continuous, 3, padding=1, padding_mode='reflect')
        else:
            self.to_latent_continuous = None
        self.quant = AdaptiveBitwiseSign()

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.stem(x)
        x = self.initial_blocks(x)
        for downsample, blocks in self.stages:
            x = downsample(x)
            x = blocks(x)
        z_discrete = self.quant(self.to_latent_discrete(x))
        if self.to_latent_continuous is not None:
            z_continuous = self.to_latent_continuous(x)
        else:
            z_continuous = torch.zeros((x.shape[0], 0, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        return z_discrete, z_continuous


class MultiScaleDecoder(nn.Module):
    def __init__(self, patch=4, dims=[128, 256, 512, 1024], depths=[4, 4, 4, 4], latent_discrete=128, latent_continuous=64):
        super().__init__()
        self.latent_continuous = latent_continuous
        total_latent = latent_discrete + latent_continuous
        self.from_latent = nn.Conv2d(total_latent, dims[-1], 1)
        self.initial_blocks = nn.Sequential(*[InceptionNeXtBlock(dims[-1]) for _ in range(depths[-1])], *[TransformerBlock(dims[-1], num_heads=16) for _ in range(8)])
        self.stages = nn.ModuleList()
        for i in range(len(dims)-1, 0, -1):
            upsample = nn.Sequential(nn.Conv2d(dims[i], dims[i-1] * 4, 1), nn.PixelShuffle(2))
            blocks = nn.Sequential(*[InceptionNeXtBlock(dims[i-1]) for _ in range(depths[i-1])])
            self.stages.append(nn.ModuleList([upsample, blocks]))
        self.to_pixels = nn.Conv2d(dims[0], 3 * patch * patch, 3, padding=1, padding_mode='reflect')
        self.shuffle = nn.PixelShuffle(patch)

    def forward(self, z_discrete, z_continuous):
        if self.latent_continuous > 0:
            z = torch.cat([z_discrete, z_continuous], dim=1)
        else:
            z = z_discrete
        x = self.from_latent(z)
        x = self.initial_blocks(x)
        for upsample, blocks in self.stages:
            x = upsample(x)
            x = blocks(x)
        img = self.shuffle(self.to_pixels(x))
        return img


class BinaryAutoencoder(nn.Module):
    def __init__(self, latent_discrete=256, latent_continuous=32):
        super().__init__()
        self.encoder = MultiScaleEncoder(latent_discrete=latent_discrete, latent_continuous=latent_continuous)
        self.decoder = MultiScaleDecoder(latent_discrete=latent_discrete, latent_continuous=latent_continuous)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z_discrete, z_continuous):
        recon = self.decoder(z_discrete, z_continuous)
        return torch.clamp(recon, -1, 1)

    def forward(self, x):
        z_discrete, z_continuous = self.encode(x)
        recon = self.decode(z_discrete, z_continuous)
        return recon, z_discrete, z_continuous


# ==========================================================
# Categorical Wrapper for AR
# ==========================================================

class CategoricalVAE(nn.Module):
    """
    Wrapper for BinaryAutoencoder to handle categorical indices for AR.
    Now just maps {-1, 0, 1} -> {0, 1, 2} for the 256 channels.
    """
    def __init__(self, latent_discrete=256, latent_continuous=32):
        super().__init__()
        self.vae = BinaryAutoencoder(latent_discrete=latent_discrete, latent_continuous=latent_continuous)
        self.latent_discrete = latent_discrete
        self.latent_continuous = latent_continuous

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) in range [-1, 1]
        Returns: (B, H/32, W/32, 256) with values in {0, 1, 2}
        """
        z_discrete, _ = self.vae.encode(x)
        # B, C, H, W -> B, H, W, C
        z_discrete = z_discrete.permute(0, 2, 3, 1).contiguous()
        # Map {-1, 0, 1} -> {0, 1, 2}
        indices = (z_discrete + 1).long()
        return indices

    def decode_from_indices(self, indices: torch.Tensor, z_continuous: torch.Tensor = None) -> torch.Tensor:
        """
        indices: (B, H/32, W/32, 256) with values in {0, 1, 2}
        """
        # Map {0, 1, 2} -> {-1, 0, 1}
        z_discrete = indices.float() - 1.0
        z_discrete = z_discrete.permute(0, 3, 1, 2).contiguous()
        
        if z_continuous is None:
            z_continuous = torch.zeros(
                (indices.shape[0], self.latent_continuous, indices.shape[1], indices.shape[2]),
                device=indices.device, dtype=z_discrete.dtype
            )
            
        return self.vae.decode(z_discrete, z_continuous)

    def load_pretrained(self, path: str, device: str = 'cpu'):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        self.vae.load_state_dict(state_dict, strict=False)
        self.eval()
