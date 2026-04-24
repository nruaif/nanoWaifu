import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ==========================================================
# Transformer Block
# ==========================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        hidden = int(dim * mlp_ratio)

        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ==========================================================
# Conv Residual MBConv-style block
# ==========================================================
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

        # Surrogate gradient: sum of sigmoid derivatives at the two thresholds
        # Scaled by 1/temp so the area under the gradient curve remains constant
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
# Ternary token bottleneck
# ==========================================================
class QuantizedTokenBottleneck(nn.Module):
    def __init__(self, dim_in, dim_latent, initial_temp=1.0):
        super().__init__()

        self.proj_in = nn.Linear(dim_in, dim_latent)
        self.norm1 = nn.LayerNorm(dim_latent)

        self.proj_out = nn.Linear(dim_latent, dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.quant = AdaptiveBitwiseSign(initial_temp=initial_temp)

    def forward(self, x):
        z = self.norm1(self.proj_in(x))
        z_q = self.quant(z)
        out = self.norm2(self.proj_out(z_q))
        return out, z_q


# ==========================================================
# Main Model
# ==========================================================
class SimpleAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dims=(64, 128, 256, 512, 512),
        expand_ratio=4,
        num_transformer_blocks=4,
        latent_dim=16,
        num_cls_tokens=4,
        num_heads=8,
        checkpoint_blocks=True,
        mask_ratio=0.75,
        mask_patch=2,
        use_masking=True,
        initial_temp=1.0,
    ):
        super().__init__()

        self.num_cls_tokens = num_cls_tokens
        self.checkpoint_blocks = checkpoint_blocks
        self.mask_ratio = mask_ratio
        self.mask_patch = mask_patch
        self.use_masking = use_masking

        # --------------------------------------------------
        # Initial downsample x2
        # --------------------------------------------------
        self.conv_in = nn.Conv2d(
            in_channels,
            hidden_dims[0],
            kernel_size=2,
            stride=2
        )

        # --------------------------------------------------
        # CNN Encoder
        # Total downsample = 32x
        # --------------------------------------------------
        enc = []
        c = hidden_dims[0]

        for h in hidden_dims[1:]:
            enc.extend([
                nn.PixelUnshuffle(2),

                nn.Conv2d(c * 4, h, 3, padding=1, bias=False),
                nn.BatchNorm2d(h),
                nn.SiLU(),

                ResnetInvertedBottleneckBlock(h, expand_ratio),
                ResnetInvertedBottleneckBlock(h, expand_ratio),

            ])
            c = h

        self.encoder = nn.Sequential(*enc)
        self.token_dim = c

        # --------------------------------------------------
        # Tokens
        # --------------------------------------------------
        self.cls_tokens = nn.Parameter(
            torch.randn(1, num_cls_tokens, c)
        )

        self.mask_token = nn.Parameter(
            torch.randn(1, 1, c)
        )

        # --------------------------------------------------
        # Transformer stacks
        # --------------------------------------------------
        self.encoder_transformers = nn.ModuleList([
            TransformerBlock(c, num_heads)
            for _ in range(num_transformer_blocks)
        ])

        self.decoder_transformers = nn.ModuleList([
            TransformerBlock(c, num_heads)
            for _ in range(num_transformer_blocks)
        ])

        # --------------------------------------------------
        # Bottleneck
        # --------------------------------------------------
        self.bottleneck = QuantizedTokenBottleneck(
            dim_in=c,
            dim_latent=latent_dim,
            initial_temp=initial_temp
        )

        # --------------------------------------------------
        # CNN Decoder
        # --------------------------------------------------
        dec = []
        rev = list(hidden_dims[::-1])

        for i in range(len(rev) - 1):
            in_dim = rev[i]
            out_dim = rev[i + 1]

            dec.extend([
                nn.Conv2d(in_dim, out_dim * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_dim * 4),
                nn.SiLU(),

                nn.PixelShuffle(2),

                ResnetInvertedBottleneckBlock(out_dim, expand_ratio),
                ResnetInvertedBottleneckBlock(out_dim, expand_ratio),

            ])

        self.decoder = nn.Sequential(*dec)

        self.conv_out = nn.ConvTranspose2d(
            hidden_dims[0],
            in_channels,
            kernel_size=2,
            stride=2
        )

    # ======================================================
    # checkpoint helper
    # ======================================================
    def run_blocks(self, x, blocks):
        for block in blocks:
            if self.training and self.checkpoint_blocks:
                x = cp.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    # ======================================================
    # token masking
    # always mask half batch during training
    # ======================================================
    def apply_masking(self, tokens, B, C, H, W):
        if (not self.training) or B < 2 or (not self.use_masking):
            return tokens

        num_masked = B // 2

        # random half of batch
        idx = torch.randperm(B, device=tokens.device)
        masked_ids = idx[:num_masked]

        cls_part = tokens[:, :self.num_cls_tokens]
        spatial = tokens[:, self.num_cls_tokens:]

        fmap = spatial.transpose(1, 2).reshape(B, C, H, W)

        p = self.mask_patch

        coarse_h = H // p
        coarse_w = W // p

        mask = (
            torch.rand(
                num_masked,
                1,
                coarse_h,
                coarse_w,
                device=tokens.device
            ) < self.mask_ratio
        )

        mask = mask.repeat_interleave(p, 2).repeat_interleave(p, 3)

        target = fmap[masked_ids]

        mask_tok = self.mask_token.view(1, C, 1, 1)
        mask_tok = mask_tok.expand_as(target)

        target = torch.where(mask, mask_tok, target)
        fmap[masked_ids] = target

        spatial = fmap.flatten(2).transpose(1, 2)
        return torch.cat([cls_part, spatial], dim=1)

    # ======================================================
    # forward
    # ======================================================
    def forward(self, x):
        # ---------------- CNN encode ----------------
        x = self.conv_in(x)
        z = self.encoder(x)

        B, C, H, W = z.shape

        # ---------------- tokens ----------------
        spatial = z.flatten(2).transpose(1, 2)
        cls = self.cls_tokens.expand(B, -1, -1)

        tokens = torch.cat([cls, spatial], dim=1)

        # ---------------- encoder transformer ----------------
        tokens = self.run_blocks(tokens, self.encoder_transformers)

        # ---------------- bottleneck ----------------
        tokens_q, latent = self.bottleneck(tokens)

        # ---------------- masking ----------------
        tokens_q = self.apply_masking(tokens_q, B, C, H, W)

        # ---------------- decoder transformer ----------------
        tokens = self.run_blocks(tokens_q, self.decoder_transformers)

        # ---------------- remove cls ----------------
        spatial = tokens[:, self.num_cls_tokens:]
        z = spatial.transpose(1, 2).reshape(B, C, H, W)

        # ---------------- decode ----------------
        x = self.decoder(z)
        x = self.conv_out(x)

        return x, latent

    def anneal_temperature(self, factor=0.98, min_temp=0.01):
        """Anneal the quantization temperature for gradual sharpening."""
        self.bottleneck.quant.anneal_temp(factor=factor, min_temp=min_temp)


# ==========================================================
# test
# ==========================================================
if __name__ == "__main__":
    model = SimpleAutoencoder().cuda()
    model.train()

    x = torch.randn(8, 3, 256, 256).cuda()

    y, latent = model(x)

    print("input :", x.shape)
    print("recon :", y.shape)
    print("latent:", latent.shape)