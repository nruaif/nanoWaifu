import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        # x: (B, L, C)

        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class ResnetInvertedBottleneckBlock(nn.Module):
    def __init__(self, channels, expand_ratio=4):
        super().__init__()

        hidden = channels * expand_ratio

        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),

            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class QuantizedTokenBottleneck(nn.Module):
    """
    Operates on token sequences:
        input  : (B, L, C)
        latent : (B, L, D)
        output : (B, L, C)

    Includes BOTH cls tokens and spatial tokens.
    """

    def __init__(self, dim_in, dim_latent):
        super().__init__()

        self.proj_in = nn.Linear(dim_in, dim_latent)
        self.proj_out = nn.Linear(dim_latent, dim_in)

    def forward(self, x):
        # x: (B, L, C)

        z = self.proj_in(x)

        # binary quantization with STE
        q = (torch.sign(z) + 1.0) / 2.0
        z_q = z + (q - z).detach()

        out = self.proj_out(z_q)

        return out, z_q


class SimpleAutoencoder(nn.Module):
    """
    CNN encoder -> Transformer -> Quantized token bottleneck
    -> Transformer -> CNN decoder

    Both CLS tokens and spatial tokens pass through bottleneck.
    """

    def __init__(
        self,
        in_channels=3,
        hidden_dims=[64, 128, 256, 512, 512],
        expand_ratio=4,
        num_transformer_blocks=4,
        latent_dim=16,
        num_cls_tokens=4,
        num_heads=8
    ):
        super().__init__()

        self.num_cls_tokens = num_cls_tokens

        # -------------------------
        # Initial downsample x2
        # -------------------------
        self.conv_in = nn.Conv2d(
            in_channels,
            hidden_dims[0],
            kernel_size=2,
            stride=2
        )

        # -------------------------
        # Encoder
        # total downsample = 32x
        # -------------------------
        encoder_layers = []

        current_channels = hidden_dims[0]

        for h_dim in hidden_dims[1:]:
            encoder_layers.extend([
                nn.PixelUnshuffle(2),

                nn.Conv2d(
                    current_channels * 4,
                    h_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(h_dim),
                nn.SiLU(),

                ResnetInvertedBottleneckBlock(h_dim, expand_ratio),
                ResnetInvertedBottleneckBlock(h_dim, expand_ratio),
            ])

            current_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # -------------------------
        # CLS + Mask token
        # -------------------------
        self.cls_tokens = nn.Parameter(
            torch.randn(1, num_cls_tokens, current_channels)
        )

        self.mask_token = nn.Parameter(
            torch.randn(1, 1, current_channels)
        )

        # -------------------------
        # Transformer encoder
        # -------------------------
        self.encoder_transformers = nn.Sequential(*[
            TransformerBlock(
                dim=current_channels,
                num_heads=num_heads
            )
            for _ in range(num_transformer_blocks)
        ])

        # -------------------------
        # Token bottleneck
        # -------------------------
        self.bottleneck = QuantizedTokenBottleneck(
            dim_in=current_channels,
            dim_latent=latent_dim
        )

        # -------------------------
        # Transformer decoder
        # -------------------------
        self.decoder_transformers = nn.Sequential(*[
            TransformerBlock(
                dim=current_channels,
                num_heads=num_heads
            )
            for _ in range(num_transformer_blocks)
        ])

        # -------------------------
        # CNN decoder
        # -------------------------
        decoder_layers = []

        rev = hidden_dims[::-1]

        for i in range(len(rev) - 1):
            in_dim = rev[i]
            out_dim = rev[i + 1]

            decoder_layers.extend([
                nn.Conv2d(
                    in_dim,
                    out_dim * 4,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_dim * 4),
                nn.SiLU(),

                nn.PixelShuffle(2),

                ResnetInvertedBottleneckBlock(out_dim, expand_ratio),
                ResnetInvertedBottleneckBlock(out_dim, expand_ratio),
            ])

        self.decoder = nn.Sequential(*decoder_layers)

        # final x2 upsample
        self.conv_out = nn.ConvTranspose2d(
            hidden_dims[0],
            in_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        # ---------------------------------
        # CNN encode
        # ---------------------------------
        x0 = self.conv_in(x)
        z = self.encoder(x0)

        # z: (B, C, H, W)
        B, C, H, W = z.shape

        # ---------------------------------
        # Flatten spatial -> tokens
        # ---------------------------------
        spatial_tokens = z.flatten(2).transpose(1, 2)   # (B, HW, C)

        cls = self.cls_tokens.expand(B, -1, -1)         # (B, T, C)

        tokens = torch.cat([cls, spatial_tokens], dim=1)  # (B, T+HW, C)

        # ---------------------------------
        # Encoder transformer
        # ---------------------------------
        tokens = self.encoder_transformers(tokens)

        # ---------------------------------
        # Quantized bottleneck
        # BOTH cls + spatial tokens
        # ---------------------------------
        tokens_q, latent = self.bottleneck(tokens)

        # latent: (B, T+HW, latent_dim)

        # ---------------------------------
        # Optional token masking
        # only mask spatial tokens
        # ---------------------------------
        if self.training and torch.rand(1).item() < 0.5:

            cls_part = tokens_q[:, :self.num_cls_tokens, :]
            spatial_part = tokens_q[:, self.num_cls_tokens:, :]

            spatial_map = spatial_part.transpose(1, 2).reshape(B, C, H, W)

            mask = torch.rand(
                B, 1, H // 2, W // 2,
                device=x.device
            ) < 0.75

            mask = mask.repeat_interleave(2, dim=2)
            mask = mask.repeat_interleave(2, dim=3)

            mask_tok = self.mask_token.view(1, C, 1, 1).expand(B, C, H, W)

            spatial_map = torch.where(mask, mask_tok, spatial_map)

            spatial_part = spatial_map.flatten(2).transpose(1, 2)

            tokens_q = torch.cat([cls_part, spatial_part], dim=1)

        # ---------------------------------
        # Decoder transformer
        # ---------------------------------
        tokens_dec = self.decoder_transformers(tokens_q)

        # ---------------------------------
        # Remove cls tokens
        # ---------------------------------
        spatial_dec = tokens_dec[:, self.num_cls_tokens:, :]  # (B, HW, C)

        z_dec = spatial_dec.transpose(1, 2).reshape(B, C, H, W)

        # ---------------------------------
        # CNN decode
        # ---------------------------------
        x_rec = self.decoder(z_dec)
        x_rec = self.conv_out(x_rec)

        return x_rec, latent


if __name__ == "__main__":
    model = SimpleAutoencoder()

    x = torch.randn(1, 3, 256, 256)

    x_rec, latent = model(x)
    print(latent)
    print("input :", x.shape)
    print("recon :", x_rec.shape)
    print("latent:", latent.shape)