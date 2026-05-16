import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# Positional Embeddings
# ==========================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

import numpy as np

def apply_rotary_emb(x, freqs_cis):
    # x: [B, N, H_heads, C]
    # freqs_cis: [N, C/2]
    x_shape = x.shape
    x = x.reshape(*x_shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x)
    
    freqs_cis = freqs_cis.view(1, x_shape[1], 1, -1) # [1, N, 1, C/2]
    
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated)
    return x_out.reshape(*x_shape)

def compute_2d_freqs_cis(dim, h, w):
    # dim is the hidden dimension of the head.
    assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
    half_dim = dim // 2
    
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    
    freqs_h = 1.0 / (10000 ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
    freqs_w = 1.0 / (10000 ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
    
    # Outer products
    freqs_h = torch.outer(grid_h, freqs_h)  # [H, D/4]
    freqs_w = torch.outer(grid_w, freqs_w)  # [W, D/4]
    
    freqs_h = torch.polar(torch.ones_like(freqs_h), freqs_h)
    freqs_w = torch.polar(torch.ones_like(freqs_w), freqs_w)
    
    # Broadcast to [H, W, D/4]
    freqs_h = freqs_h.unsqueeze(1).expand(h, w, -1)
    freqs_w = freqs_w.unsqueeze(0).expand(h, w, -1)
    
    # Concatenate along dim
    freqs_cis = torch.cat([freqs_h, freqs_w], dim=-1) # [H, W, D/2] complex
    return freqs_cis.reshape(h * w, -1)

# ==========================================================
# Blocks
# ==========================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNZero(nn.Module):
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond):
        emb = self.linear(self.silu(cond))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AttentionRoPE(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # [B, H, N, D]

        if freqs_cis is not None:
            # Apply RoPE
            q_roped = apply_rotary_emb(q.transpose(1, 2), freqs_cis).transpose(1, 2)
            k_roped = apply_rotary_emb(k.transpose(1, 2), freqs_cis).transpose(1, 2)
            q, k = q_roped, k_roped

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        x = self.fc1(x)
        
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlockAdaLN(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, cond_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionRoPE(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.adaLN = AdaLNZero(cond_dim, dim) if cond_dim else None

    def forward(self, x, cond, freqs_cis=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond)
        
        # Attention
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm_x, freqs_cis)
        
        # MLP
        norm_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x)
        return x

class MLPMixerBlock(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_tokens, num_tokens),
            nn.GELU(),
            nn.Linear(num_tokens, num_tokens)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x: [B, N, C]
        h = self.norm1(x)
        h = h.transpose(1, 2) # [B, C, N]
        h = self.token_mix(h)
        h = h.transpose(1, 2) # [B, N, C]
        x = x + h
        
        x = x + self.channel_mix(self.norm2(x))
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, N, C] or [B, N]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def spherify(z, sampling=False, sigma=None, max_sigma=None, e=None):
    """
    RMS normalize z to have average magnitude 1, meaning L2 norm is sqrt(L).
    v = z / sqrt(mean(z^2))
    """
    L = z.numel() // z.shape[0]
    
    def normalize(v):
        var = v.pow(2).mean(dim=[1, 2], keepdim=True) if v.ndim == 3 else v.pow(2).mean(dim=-1, keepdim=True)
        return v * torch.rsqrt(var + 1e-6)

    v = normalize(z)
    
    if sampling and sigma is not None:
        if e is None:
            e = torch.randn_like(v)
        v = normalize(v + sigma * e)
        
    return v

# ==========================================================
# Main Models
# ==========================================================

class SphereEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_chans=3, dim=1024, depth=24, num_heads=16, 
                 mlp_ratio=4.0, cond_dim=1024, mixer_depth=4, latent_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        
        # Absolute positional embeddings
        pos_embed = get_2d_sincos_pos_embed(dim, self.grid_size)
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        
        # RoPE
        freqs_cis = compute_2d_freqs_cis(dim // num_heads, self.grid_size, self.grid_size)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        
        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(dim, num_heads, mlp_ratio, cond_dim)
            for _ in range(depth)
        ])
        
        self.mixers = nn.ModuleList([
            MLPMixerBlock(self.num_patches, dim)
            for _ in range(mixer_depth)
        ])
        
        self.to_latent = nn.Linear(dim, latent_dim)
        self.rms_norm = RMSNorm(latent_dim)

    def forward(self, x, cond):
        # x: [B, C, H, W]
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, N, C]
        
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x, cond, self.freqs_cis)
            
        for mixer in self.mixers:
            x = mixer(x)
            
        x = self.to_latent(x) # [B, N, latent_dim]
        x = self.rms_norm(x)
        return x

class SphereDecoder(nn.Module):
    def __init__(self, img_size=256, patch_size=8, out_chans=3, dim=1024, depth=24, num_heads=16, 
                 mlp_ratio=4.0, cond_dim=1024, mixer_depth=4, latent_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        self.rms_norm = RMSNorm(latent_dim)
        self.from_latent = nn.Linear(latent_dim, dim)
        
        self.mixers = nn.ModuleList([
            MLPMixerBlock(self.num_patches, dim)
            for _ in range(mixer_depth)
        ])
        
        # Absolute positional embeddings
        pos_embed = get_2d_sincos_pos_embed(dim, self.grid_size)
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        
        # RoPE
        freqs_cis = compute_2d_freqs_cis(dim // num_heads, self.grid_size, self.grid_size)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        
        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(dim, num_heads, mlp_ratio, cond_dim)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = AdaLNZero(cond_dim, dim)
        
        self.unpatchify = nn.Linear(dim, patch_size * patch_size * out_chans)

    def forward(self, x, cond):
        # x: [B, N, latent_dim]
        x = self.rms_norm(x)
        x = self.from_latent(x)
        
        for mixer in self.mixers:
            x = mixer(x)
            
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x, cond, self.freqs_cis)
            
        # Final norm
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond)
        x = modulate(self.norm(x), shift_msa, scale_msa)
        
        x = self.unpatchify(x) # [B, N, P*P*C]
        
        # Unpatchify
        B = x.shape[0]
        P = self.patch_size
        H = W = self.grid_size
        C = x.shape[-1] // (P * P)
        
        x = x.reshape(B, H, W, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H * P, W * P)
        return x

class TagConditioner(nn.Module):
    def __init__(self, vocab_size, cond_dim, depth=2, num_heads=8):
        super().__init__()
        self.tag_embed = nn.Embedding(vocab_size, cond_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cond_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cond_dim, 
            nhead=num_heads, 
            dim_feedforward=int(cond_dim * 4), 
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, tags):
        # tags: [B, SeqLen]
        B = tags.shape[0]
        x = self.tag_embed(tags) # [B, SeqLen, cond_dim]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, SeqLen + 1, cond_dim]
        
        # 2 transformer blocks, no pos encoding
        x = self.transformer(x)
        
        # Extract cls token
        cond = x[:, 0, :] # [B, cond_dim]
        return cond


class SphereAutoencoder(nn.Module):
    def __init__(self, vocab_size=10000, img_size=256, patch_size=8, dim=1024, enc_depth=24, dec_depth=24, 
                 latent_dim=128, num_heads=16):
        super().__init__()
        self.vocab_size = vocab_size
        cond_dim = dim
        
        # Tag conditioning via 2 transformer blocks
        self.tag_conditioner = TagConditioner(vocab_size, cond_dim, depth=2, num_heads=num_heads)
        # Unconditional (null) embedding for CFG
        self.null_emb = nn.Parameter(torch.randn(cond_dim))
        
        self.encoder = SphereEncoder(
            img_size=img_size, patch_size=patch_size, dim=dim, depth=enc_depth, num_heads=num_heads,
            cond_dim=cond_dim, latent_dim=latent_dim
        )
        self.decoder = SphereDecoder(
            img_size=img_size, patch_size=patch_size, dim=dim, depth=dec_depth, num_heads=num_heads,
            cond_dim=cond_dim, latent_dim=latent_dim
        )

    def get_cond(self, tags, cfg_prob=0.1):
        B = tags.shape[0]
        cond = self.tag_conditioner(tags)
        if self.training and cfg_prob > 0.0:
            mask = torch.rand(B, device=tags.device) < cfg_prob
            cond[mask] = self.null_emb
        return cond

    def forward(self, x, tags, noise_r=1.0, max_sigma=28.6):
        cond = self.get_cond(tags)
        z = self.encoder(x, cond)
        
        # Spherify with jitter
        sigma = noise_r * max_sigma
        
        # Sub-sigma for small jitter (pixel reconstruction)
        s = torch.rand(x.shape[0], 1, 1, device=x.device) * 0.5
        sigma_sub = s * sigma
        
        # Shared noise direction e
        e = torch.randn_like(z)
        
        # Latents
        v = spherify(z, sampling=False)
        v_noisy = spherify(z, sampling=True, sigma=sigma_sub, e=e)
        v_NOISY = spherify(z, sampling=True, sigma=sigma, e=e)
        
        # Decode
        recon_noisy = self.decoder(v_noisy, cond)
        recon_NOISY = self.decoder(v_NOISY, cond)
        
        return recon_noisy, recon_NOISY, v, cond

    def generate(self, e, tags=None, steps=1, cfg_scale=1.0, max_sigma=28.6):
        B = e.shape[0]
        device = e.device
        
        if tags is None:
            cond = self.null_emb.unsqueeze(0).expand(B, -1)
            cond_null = cond
        else:
            cond = self.tag_conditioner(tags)
            cond_null = self.null_emb.unsqueeze(0).expand(B, -1)
            
        v = spherify(e, sampling=False)
        x = self.decoder(v, cond)
        
        if cfg_scale > 1.0 and tags is not None:
            x_uncond = self.decoder(v, cond_null)
            x = x_uncond + cfg_scale * (x - x_uncond)
            
        if steps > 1:
            for _ in range(steps - 1):
                z = self.encoder(x, cond)
                if cfg_scale > 1.0 and tags is not None:
                    z_uncond = self.encoder(x, cond_null)
                    z = z_uncond + cfg_scale * (z - z_uncond)
                
                v = spherify(z, sampling=True, sigma=max_sigma)
                x = self.decoder(v, cond)
                
                if cfg_scale > 1.0 and tags is not None:
                    x_uncond = self.decoder(v, cond_null)
                    x = x_uncond + cfg_scale * (x - x_uncond)
                    
        return x

if __name__ == "__main__":
    model = SphereAutoencoder(vocab_size=1000, img_size=32, patch_size=2, dim=256, enc_depth=4, dec_depth=4, latent_dim=8)
    x = torch.randn(2, 3, 32, 32)
    tags = torch.randint(0, 1000, (2, 16))
    
    recon_noisy, recon_NOISY, v, cond = model(x, tags)
    print("Recon shape:", recon_noisy.shape)
    
    # Test generation
    e = torch.randn(2, 16*16, 8)
    gen = model.generate(e, tags, steps=2)
    print("Gen shape:", gen.shape)
