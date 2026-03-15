import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def relu2(x):
    """ReLU^2 activation function."""
    return torch.square(F.relu(x))

class GRN(nn.Module):
    """Global Response Normalization [cite: 64, 101]"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class FCDMBlock(nn.Module):
    """Standard ConvNeXt-like block for FCDM with AdaLN and ReLU^2."""
    def __init__(self, dim, cond_dim, r=3):
        super().__init__()
        # 7x7 Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # 1x1 Pointwise convolutions with expansion ratio r
        self.pwconv1 = nn.Conv2d(dim, dim * r, kernel_size=1)
        self.grn = GRN(dim * r)
        self.pwconv2 = nn.Conv2d(dim * r, dim, kernel_size=1)

        # MLP for AdaLN modulation (shift, scale, gate/alpha)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shortcut = x
        x = self.dwconv(x)
        
        # LayerNorm expects channel last
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        # AdaLN Shift and Scale
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        shift, scale, gate = shift[..., None, None], scale[..., None, None], gate[..., None, None]

        x = x * (1 + scale) + shift
        
        # Inverted bottleneck
        x = self.pwconv1(x)
        x = relu2(x) # ReLU^2
        x = self.grn(x)
        x = self.pwconv2(x)
        
        # Gate/Alpha scaling and skip connection
        return shortcut + x * gate

class ViTBlock(nn.Module):
    """Vision Transformer block with AdaLN and ReLU^2."""
    def __init__(self, dim, cond_dim, r=4):
        super().__init__()
        # Ensure dim is divisible by num_heads. 
        num_heads = 8
        if dim % num_heads != 0:
            for h in [8, 4, 2, 1]:
                if dim % h == 0:
                    num_heads = h
                    break
        
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * r),
            # ReLU^2 handled in forward
            nn.Linear(dim * r, dim)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim) # 3 for attn, 3 for mlp
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # AdaLN modulation
        mods = self.adaLN_modulation(c).chunk(6, dim=1)
        shift1, scale1, gate1, shift2, scale2, gate2 = [m[:, None, :] for m in mods]

        # Attention Branch
        shortcut = x_flat
        x_norm = self.norm1(x_flat)
        x_norm = x_norm * (1 + scale1) + shift1
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        # ReLU^2 not usually in attention, but let's check: request said "All activativation use ReLU ^2"
        # Usually attn uses softmax. We'll stick to standard attn but ReLU^2 in MLP.
        x_flat = shortcut + x_attn * gate1

        # MLP Branch
        shortcut = x_flat
        x_norm = self.norm2(x_flat)
        x_norm = x_norm * (1 + scale2) + shift2
        
        x_mlp = self.mlp[0](x_norm)
        x_mlp = relu2(x_mlp)
        x_mlp = self.mlp[1](x_mlp)
        
        x_flat = shortcut + x_mlp * gate2
        
        return x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

class CSPStage(nn.Module):
    """Cross Stage Partial Stage.
    Switches to a ViT block every 3 blocks."""
    def __init__(self, dim, cond_dim, num_blocks):
        super().__init__()
        self.split_dim = dim // 2
        
        blocks = []
        for i in range(1, num_blocks + 1):
            if i % 3 == 0:
                blocks.append(ViTBlock(self.split_dim, cond_dim))
            else:
                blocks.append(FCDMBlock(self.split_dim, cond_dim))
        
        self.blocks = nn.ModuleList(blocks)
        # Fusion layer to mix the partial branches
        self.conv_fuse = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, c, feat_layers=None, start_idx=0):
        # Split channels: x1 is identity, x2 goes through blocks
        x1, x2 = x.chunk(2, dim=1)
        
        feats = {}
        for i, block in enumerate(self.blocks):
            x2 = block(x2, c)
            if feat_layers and (start_idx + i + 1) in feat_layers:
                # Return fused intermediate features if requested
                feats[start_idx + i + 1] = self.conv_fuse(torch.cat([x1, x2], dim=1))
        
        # Concatenate and fuse
        out = self.conv_fuse(torch.cat([x1, x2], dim=1))
        if feat_layers:
            return out, feats
        return out

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hidden_dim = hidden_dim

    def forward(self, t):
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)

class FCDMV2(nn.Module):
    """Fully Convolutional Diffusion Model U-Net V2 with CSP, Hybrid ViT, ReLU^2 and PixelShuffle Head."""
    def __init__(self, in_channels=3, base_channels=128, num_blocks=2, num_classes=12476, patch_size=16, use_t_cond=True):
        super().__init__()
        self.c = base_channels
        self.l = num_blocks
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.use_t_cond = use_t_cond
        
        if self.use_t_cond:
            self.t_embedder = TimestepEmbedder(self.c * 4)
        self.y_embedder = nn.EmbeddingBag(num_classes + 1, self.c * 4, mode='mean')
        cond_dim = self.c * 4

        # Input Convolution with PixelUnshuffle
        self.unshuffle = nn.PixelUnshuffle(patch_size)
        self.conv_in = nn.Conv2d(in_channels * (patch_size ** 2), self.c, kernel_size=7, padding=3)

        # Stage 1 (Encoder)
        self.enc1 = CSPStage(self.c, cond_dim, self.l)
        self.down1 = nn.Conv2d(self.c, self.c * 2, kernel_size=2, stride=2)

        # Stage 2 (Encoder)
        self.enc2 = CSPStage(self.c * 2, cond_dim, self.l * 2)
        self.down2 = nn.Conv2d(self.c * 2, self.c * 4, kernel_size=2, stride=2)

        # Stage 3 (Bottleneck)
        self.mid = CSPStage(self.c * 4, cond_dim, self.l * 4)

        # Stage 2 (Decoder)
        self.up2 = nn.ConvTranspose2d(self.c * 4, self.c * 2, kernel_size=2, stride=2)
        self.skip_proj2 = nn.Conv2d(self.c * 4, self.c * 2, kernel_size=1)
        self.dec2 = CSPStage(self.c * 2, cond_dim, self.l * 2)

        # Stage 1 (Decoder)
        self.up1 = nn.ConvTranspose2d(self.c * 2, self.c, kernel_size=2, stride=2)
        self.skip_proj1 = nn.Conv2d(self.c * 2, self.c, kernel_size=1)
        self.dec1 = CSPStage(self.c, cond_dim, self.l)

        # Output Head: Simple "Unconv" (PixelShuffle)
        self.norm_out = nn.LayerNorm(self.c * 2)
        self.conv_out = nn.Conv2d(self.c * 2, in_channels * (patch_size ** 2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x, t, y_indices, y_offsets=None, feat_layers=None):
        c = self.y_embedder(y_indices, y_offsets)
        if self.use_t_cond:
            c = c + self.t_embedder(t)
        
        # Pixel Unshuffle and project
        x = self.unshuffle(x)
        x = self.conv_in(x)
        skip_in = x # Final skip connection
        
        feats = {}
        # Encoder
        if feat_layers:
            x, f = self.enc1(x, c, feat_layers, start_idx=0)
            feats.update(f)
        else: x = self.enc1(x, c)
        
        skip1 = x
        x = self.down1(x)
        
        if feat_layers:
            x, f = self.enc2(x, c, feat_layers, start_idx=self.l)
            feats.update(f)
        else: x = self.enc2(x, c)
        
        skip2 = x
        x = self.down2(x)
        
        # Mid
        if feat_layers:
            x, f = self.mid(x, c, feat_layers, start_idx=self.l + self.l * 2)
            feats.update(f)
        else: x = self.mid(x, c)
            
        # Decoder
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.skip_proj2(x)
        if feat_layers:
            x, f = self.dec2(x, c, feat_layers, start_idx=self.l + self.l * 2 + self.l * 4)
            feats.update(f)
        else: x = self.dec2(x, c)
            
        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.skip_proj1(x)
        if feat_layers:
            x, f = self.dec1(x, c, feat_layers, start_idx=self.l + self.l * 2 + self.l * 4 + self.l * 2)
            feats.update(f)
        else: x = self.dec1(x, c)
            
        # Output Head
        x = torch.cat([x, skip_in], dim=1) # Concat global skip connection
        x = x.permute(0, 2, 3, 1)
        x = self.norm_out(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_out(x)
        x = self.shuffle(x)
        
        if feat_layers:
            return x, feats
        return x

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

@torch.no_grad()
def sample_flow(model, tag_processor, image_size, batch_size, prompts, device,
                steps=50, cfg_scale=1.4, noise=None):
    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    y_indices, y_offsets = tag_processor.process_prompts(prompts, device)
    null_indices = torch.full((batch_size,), fill_value=tag_processor.num_classes, dtype=torch.long, device=device)
    null_offsets = torch.arange(batch_size, dtype=torch.long, device=device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i+1]
        t_vec = torch.full((batch_size,), t_curr, device=device)
        x1_cond = model(x, t_vec, y_indices, y_offsets)
        x1_uncond = model(x, t_vec, null_indices, null_offsets)
        x0_pred = x1_uncond + cfg_scale * (x1_cond - x1_uncond)
        if t_curr > 0:
            x = (t_next / t_curr) * x + (1 - t_next / t_curr) * x0_pred
        else:
            x = x0_pred
    images = (x / 2 + 0.5).clamp(0, 1)
    return images
