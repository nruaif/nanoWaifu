import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        # x is (B, L, C)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ResnetInvertedBottleneckBlock(nn.Module):
    def __init__(self, channels, expand_ratio=4):
        super().__init__()
        hidden_dim = channels * expand_ratio

        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class QuantizedConv2dBottleneck(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, dim, 1)
        self.proj_out = nn.Conv2d(dim, in_channels, 1)

    def forward(self, x):
        # x is (B, C, H, W) — quantization operates over the channel dim as intended
        x = self.proj_in(x)

        # Binarize channels with STE: values in {0, 1}
        q = (torch.sign(x) + 1.0) / 2.0
        x_q = x + (q - x).detach()

        out = self.proj_out(x_q)
        return out, x_q


class SimpleAutoencoder(nn.Module):
    """
    A simple autoencoder using ResNet inverted bottleneck blocks and
    PixelUnshuffle/PixelShuffle for downsampling and upsampling.
    """

    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512, 512], expand_ratio=4, num_transformer_blocks=4,
                 dim=16):
        super().__init__()

        # Initial conv (2x downsampling)
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=2, stride=2)

        # Encoder (4 stages of 2x downsampling -> total 32x downsampling when combined with conv_in)
        encoder_layers = []
        current_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            encoder_layers.extend([
                nn.PixelUnshuffle(2),
                nn.Conv2d(current_channels * 4, h_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(h_dim),
                nn.SiLU(),
                ResnetInvertedBottleneckBlock(h_dim, expand_ratio),
                ResnetInvertedBottleneckBlock(h_dim, expand_ratio)
            ])
            current_channels = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Bottleneck (Conv2D -> Quantize -> Conv2D) — operates on (B, C, H, W) spatial maps
        self.bottleneck = QuantizedConv2dBottleneck(current_channels, dim)

        # CLS and Mask tokens
        self.cls_tokens = nn.Parameter(torch.randn(1, 4, current_channels))
        self.mask_token = nn.Parameter(torch.randn(1, current_channels))

        # Transformers
        encoder_transformer_layers = []
        for _ in range(num_transformer_blocks):
            encoder_transformer_layers.append(TransformerBlock(current_channels))
        self.encoder_transformers = nn.Sequential(*encoder_transformer_layers)

        decoder_transformer_layers = []
        for _ in range(num_transformer_blocks):
            decoder_transformer_layers.append(TransformerBlock(current_channels))
        self.decoder_transformers = nn.Sequential(*decoder_transformer_layers)

        # Decoder
        decoder_layers = []
        reversed_dims = hidden_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            in_dim = reversed_dims[i]
            out_dim = reversed_dims[i + 1]

            decoder_layers.extend([
                nn.Conv2d(in_dim, out_dim * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_dim * 4),
                nn.SiLU(),
                nn.PixelShuffle(2),
                ResnetInvertedBottleneckBlock(out_dim, expand_ratio),
                ResnetInvertedBottleneckBlock(out_dim, expand_ratio)
            ])

        self.decoder = nn.Sequential(*decoder_layers)

        # Final conv out (2x upsampling to reverse conv_in)
        self.conv_out = nn.ConvTranspose2d(hidden_dims[0], in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_in(x)
        z = self.encoder(x)

        B, C, H, W = z.shape

        # Flatten spatial dimensions to sequence: (B, H*W, C)
        z_seq = z.flatten(2).transpose(1, 2)

        # Prepend 4 cls tokens: (B, 4, C)
        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        z_seq = torch.cat([cls_tokens, z_seq], dim=1)

        # Pass through encoder transformers
        z_seq = self.encoder_transformers(z_seq)

        # --- Bottleneck on spatial tokens only (channel-dim quantization) ---
        # Separate cls tokens and spatial tokens before the bottleneck
        cls_enc = z_seq[:, :4, :]  # (B, 4, C)
        spatial_enc = z_seq[:, 4:, :]  # (B, H*W, C)

        # Reshape spatial tokens to (B, C, H, W) so Conv2d quantizes over channels
        z_spatial = spatial_enc.transpose(1, 2).reshape(B, C, H, W)

        # Apply bottleneck: quantization now acts on the channel dimension
        z_spatial_out, latent_spatial = self.bottleneck(z_spatial)
        # latent_spatial is (B, dim, H, W) — binary channel representation

        # Flatten bottleneck output back to sequence: (B, H*W, C)
        spatial_out = z_spatial_out.flatten(2).transpose(1, 2)

        # Re-attach cls tokens from encoder
        z_seq_dec = torch.cat([cls_enc, spatial_out], dim=1)  # (B, 4+H*W, C)

        # 50% chance to apply masking; when active, masks 75% of spatial tokens in 2x2 blocks
        if self.training and torch.rand(1).item() < 0.5:
            mask = torch.rand(B, 1, H // 2, W // 2, device=z.device) < 0.75
            mask = mask.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # (B, 1, H, W)
            mask_token = self.mask_token.view(1, C, 1, 1).expand(B, C, H, W)
            z_spatial_masked = torch.where(mask, mask_token, z_spatial_out)
            spatial_masked = z_spatial_masked.flatten(2).transpose(1, 2)
            z_seq_dec = torch.cat([cls_enc, spatial_masked], dim=1)

        # Pass through decoder transformers
        z_seq_dec = self.decoder_transformers(z_seq_dec)

        # Discard cls tokens, reshape spatial tokens to (B, C, H, W)
        spatial_out_dec = z_seq_dec[:, 4:, :]
        z_spatial_dec = spatial_out_dec.transpose(1, 2).reshape(B, C, H, W)

        x_rec = self.decoder(z_spatial_dec)
        x_rec = self.conv_out(x_rec)
        return x_rec, latent_spatial