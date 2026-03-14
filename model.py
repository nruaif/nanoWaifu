import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

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
    """Modified ConvNeXt block for FCDM with AdaLN conditional injection [cite: 66, 67]"""
    def __init__(self, dim, cond_dim, r=3):
        super().__init__()
        # 7x7 Depthwise convolution [cite: 63]
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # 1x1 Pointwise convolutions with expansion ratio r (r=3 by default) [cite: 64, 306]
        self.pwconv1 = nn.Conv2d(dim, dim * r, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(dim * r)
        self.pwconv2 = nn.Conv2d(dim * r, dim, kernel_size=1)

        # MLP for AdaLN modulation (shift, scale, gate/alpha) [cite: 67]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim)
        )
        # Zero-initialize the final modulation scale to stabilize optimization [cite: 68]
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shortcut = x
        x = self.dwconv(x)
        
        # LayerNorm expects channel last
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        # AdaLN Shift and Scale [cite: 67]
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        shift, scale, gate = shift[..., None, None], scale[..., None, None], gate[..., None, None]

        x = x * (1 + scale) + shift
        
        # Inverted bottleneck [cite: 97]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        # Gate/Alpha scaling [cite: 67]
        x = x * gate
        return x + shortcut

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

class FCDM(nn.Module):
    """Fully Convolutional Diffusion Model U-Net [cite: 71, 72]"""
    def __init__(self, in_channels=3, base_channels=128, num_blocks=2, num_classes=12476, patch_size=16):
        super().__init__()
        self.c = base_channels
        self.l = num_blocks
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Conditioning embeddings (+1 for null token)
        self.t_embedder = TimestepEmbedder(self.c * 4)
        # Using EmbeddingBag for multi-label support
        self.y_embedder = nn.EmbeddingBag(num_classes + 1, self.c * 4, mode='mean')
        cond_dim = self.c * 4

        # Input Convolution with PixelUnshuffle
        self.unshuffle = nn.PixelUnshuffle(patch_size)
        self.conv_in = nn.Conv2d(in_channels * (patch_size ** 2), self.c, kernel_size=7, padding=3)

        # Stage 1 (Encoder) [cite: 424]
        self.enc1 = nn.ModuleList([FCDMBlock(self.c, cond_dim) for _ in range(self.l)])
        self.down1 = nn.Conv2d(self.c, self.c * 2, kernel_size=2, stride=2)

        # Stage 2 (Encoder)
        self.enc2 = nn.ModuleList([FCDMBlock(self.c * 2, cond_dim) for _ in range(self.l * 2)])
        self.down2 = nn.Conv2d(self.c * 2, self.c * 4, kernel_size=2, stride=2)

        # Stage 3 (Bottleneck)
        self.mid = nn.ModuleList([FCDMBlock(self.c * 4, cond_dim) for _ in range(self.l * 4)])

        # Stage 2 (Decoder) - Concatenates skip connection
        self.up2 = nn.ConvTranspose2d(self.c * 4, self.c * 2, kernel_size=2, stride=2)
        self.skip_proj2 = nn.Conv2d(self.c * 4, self.c * 2, kernel_size=1)
        self.dec2 = nn.ModuleList([FCDMBlock(self.c * 2, cond_dim) for _ in range(self.l * 2)])

        # Stage 1 (Decoder)
        self.up1 = nn.ConvTranspose2d(self.c * 2, self.c, kernel_size=2, stride=2)
        self.skip_proj1 = nn.Conv2d(self.c * 2, self.c, kernel_size=1)
        self.dec1 = nn.ModuleList([FCDMBlock(self.c, cond_dim) for _ in range(self.l)])

        # Output
        self.norm_out = nn.LayerNorm(self.c)
        self.conv_out = nn.Conv2d(self.c, in_channels * (patch_size ** 2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x, t, y_indices, y_offsets=None):
        """
        x: (B, C, H, W)
        t: (B,)
        y_indices: (Total_Tags,) or (B, Tags_Per_Sample)
        y_offsets: (B,) if y_indices is flattened
        """
        # Conditioning
        c = self.t_embedder(t) + self.y_embedder(y_indices, y_offsets)
        
        # Pixel Unshuffle and project
        x = self.unshuffle(x)
        x = self.conv_in(x)
        
        # Stage 1
        for block in self.enc1: x = block(x, c)
        skip1 = x
        x = self.down1(x)
        
        # Stage 2
        for block in self.enc2: x = block(x, c)
        skip2 = x
        x = self.down2(x)
        
        # Stage 3
        for block in self.mid: x = block(x, c)
            
        # Stage 2 (Decode)
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1) # U-Net Skip connection
        x = self.skip_proj2(x)
        for block in self.dec2: x = block(x, c)
            
        # Stage 1 (Decode)
        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.skip_proj1(x)
        for block in self.dec1: x = block(x, c)
            
        # Output
        x = x.permute(0, 2, 3, 1)
        x = self.norm_out(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_out(x)
        x = self.shuffle(x)
        return x

class TagProcessor:
    def __init__(self, tags_file):
        with open(tags_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.num_classes = len(self.tags)
    
    def process_prompts(self, prompts, device, dropout_prob=0.0):
        # prompts is a list of strings
        indices = []
        offsets = [0]
        for p in prompts:
            if random.random() < dropout_prob:
                indices.append(self.num_classes) # Null tag
            else:
                tags = p.split()
                count = 0
                for t in tags:
                    if t in self.tag_to_idx:
                        indices.append(self.tag_to_idx[t])
                        count += 1
                if count == 0:
                    indices.append(self.num_classes) # Null tag if no match
            offsets.append(len(indices))
        
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return indices, offsets

@torch.no_grad()
def sample_flow(model, tag_processor, image_size, batch_size, prompts, device,
                steps=50, cfg_scale=1.4, noise=None):
    """
    Sample using Euler integration with true CFG for FCDM.
    Handles list of noise tensors for varying sizes.
    """
    # 1. Initialize Image Gaussian Noise
    if noise is not None:
        if isinstance(noise, list):
            x_list = [n.clone().to(device) for n in noise[:batch_size]]
        else:
            x = noise.clone().to(device)
            # If single tensor, we assume fixed size for all in batch
            x_list = [x[i] if i < x.shape[0] else x[0] for i in range(batch_size)]
    else:
        # Default to image_size x image_size if no noise provided
        x_list = [torch.randn(3, image_size, image_size, device=device) for _ in range(batch_size)]

    dt = 1.0 / steps

    # 2. Process prompts for CFG
    y_indices, y_offsets = tag_processor.process_prompts(prompts, device)
    
    # Null indices (unconditioned)
    null_indices = torch.full((batch_size,), fill_value=tag_processor.num_classes, dtype=torch.long, device=device)
    null_offsets = torch.arange(batch_size, dtype=torch.long, device=device)

    def get_v_single(x_curr, t_curr, idx):
        t_scaled = torch.full((1,), t_curr, device=device)
        
        # Conditioned forward
        start_off = y_offsets[idx].item()
        end_off = y_offsets[idx+1].item() if idx+1 < len(y_offsets) else len(y_indices)
        y_ind = y_indices[start_off:end_off]
        y_off = torch.zeros(1, dtype=torch.long, device=device)
        
        x1_cond = model(x_curr.unsqueeze(0), t_scaled, y_ind, y_off).squeeze(0)
        
        # Unconditioned forward
        y_ind_null = null_indices[idx:idx+1]
        y_off_null = torch.zeros(1, dtype=torch.long, device=device)
        x1_uncond = model(x_curr.unsqueeze(0), t_scaled, y_ind_null, y_off_null).squeeze(0)
        
        x1_cfg = x1_uncond + cfg_scale * (x1_cond - x1_uncond)
        v = (x1_cfg - x_curr) / (1.0 - t_curr + 1e-7)
        return v

    # 3. Euler Loop
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(range(steps), desc='Euler Sampling', leave=False)
    except ImportError:
        pbar = range(steps)

    for i in pbar:
        t = i / steps
        new_x_list = []
        for j, x_curr in enumerate(x_list):
            v = get_v_single(x_curr, t, j)
            new_x_list.append(x_curr + v * dt)
        x_list = new_x_list

    # Final images
    images = [(x / 2 + 0.5).clamp(0, 1) for x in x_list]
    # If all same size, stack them
    try:
        images = torch.stack(images)
    except:
        # Keep as list if different sizes (caller needs to handle)
        pass
    return images
