import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # [B, H, N, D]

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

    def forward(self, x, H, W):
        B, N, C = x.shape
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
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.adaLN = AdaLNZero(cond_dim, dim) if cond_dim else None

    def forward(self, x, cond, H, W):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond)
        
        # Attention
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm_x)
        
        # MLP
        norm_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x, H, W)
        return x

class MLPMixerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, H, W):
        # x: [B, N, C]
        B, N, C = x.shape
        h = self.norm1(x)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        h = self.token_mix(h)
        h = h.flatten(2).transpose(1, 2)
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
    def __init__(self, patch_size=8, in_chans=3, dim=1024, depth=24, num_heads=16, 
                 mlp_ratio=4.0, cond_dim=1024, mixer_depth=4, latent_dim=128):
        super().__init__()
        self.patch_size = patch_size
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        
        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(dim, num_heads, mlp_ratio, cond_dim)
            for _ in range(depth)
        ])
        
        self.mixers = nn.ModuleList([
            MLPMixerBlock(dim)
            for _ in range(mixer_depth)
        ])
        
        self.to_latent = nn.Linear(dim, latent_dim)
        self.rms_norm = RMSNorm(latent_dim)

    def forward(self, x, cond):
        # x: [B, C, H_img, W_img]
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, N, C]
        
        for block in self.blocks:
            x = block(x, cond, H, W)
            
        for mixer in self.mixers:
            x = mixer(x, H, W)
            
        x = self.to_latent(x) # [B, N, latent_dim]
        x = self.rms_norm(x)
        return x, H, W

class SphereDecoder(nn.Module):
    def __init__(self, patch_size=8, out_chans=3, dim=1024, depth=24, num_heads=16, 
                 mlp_ratio=4.0, cond_dim=1024, mixer_depth=4, latent_dim=128):
        super().__init__()
        self.patch_size = patch_size
        
        self.rms_norm = RMSNorm(latent_dim)
        self.from_latent = nn.Linear(latent_dim, dim)
        
        self.mixers = nn.ModuleList([
            MLPMixerBlock(dim)
            for _ in range(mixer_depth)
        ])
        
        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(dim, num_heads, mlp_ratio, cond_dim)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = AdaLNZero(cond_dim, dim)
        
        self.unpatchify = nn.Linear(dim, patch_size * patch_size * out_chans)

    def forward(self, x, cond, H, W):
        # x: [B, N, latent_dim]
        x = self.rms_norm(x)
        x = self.from_latent(x)
        
        for mixer in self.mixers:
            x = mixer(x, H, W)
            
        for block in self.blocks:
            x = block(x, cond, H, W)
            
        # Final norm
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond)
        x = modulate(self.norm(x), shift_msa, scale_msa)
        
        x = self.unpatchify(x) # [B, N, P*P*C]
        
        # Unpatchify
        B = x.shape[0]
        P = self.patch_size
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
    def __init__(self, vocab_size=10000, patch_size=8, dim=1024, enc_depth=24, dec_depth=24, 
                 latent_dim=128, num_heads=16):
        super().__init__()
        self.vocab_size = vocab_size
        cond_dim = dim
        self.patch_size = patch_size
        
        # Tag conditioning via 2 transformer blocks
        self.tag_conditioner = TagConditioner(vocab_size, cond_dim, depth=2, num_heads=num_heads)
        # Unconditional (null) embedding for CFG
        self.null_emb = nn.Parameter(torch.randn(cond_dim))
        
        self.encoder = SphereEncoder(
            patch_size=patch_size, dim=dim, depth=enc_depth, num_heads=num_heads,
            cond_dim=cond_dim, latent_dim=latent_dim
        )
        self.decoder = SphereDecoder(
            patch_size=patch_size, dim=dim, depth=dec_depth, num_heads=num_heads,
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
        z, H, W = self.encoder(x, cond)
        
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
        recon_noisy = self.decoder(v_noisy, cond, H, W)
        recon_NOISY = self.decoder(v_NOISY, cond, H, W)
        
        return recon_noisy, recon_NOISY, v, cond

    def generate(self, e, H, W, tags=None, steps=1, cfg_scale=1.0, max_sigma=28.6):
        B = e.shape[0]
        device = e.device
        
        if tags is None:
            cond = self.null_emb.unsqueeze(0).expand(B, -1)
            cond_null = cond
        else:
            cond = self.tag_conditioner(tags)
            cond_null = self.null_emb.unsqueeze(0).expand(B, -1)
            
        v = spherify(e, sampling=False)
        x = self.decoder(v, cond, H, W)
        
        if cfg_scale > 1.0 and tags is not None:
            x_uncond = self.decoder(v, cond_null, H, W)
            x = x_uncond + cfg_scale * (x - x_uncond)
            
        if steps > 1:
            for _ in range(steps - 1):
                z, _, _ = self.encoder(x, cond)
                if cfg_scale > 1.0 and tags is not None:
                    z_uncond, _, _ = self.encoder(x, cond_null)
                    z = z_uncond + cfg_scale * (z - z_uncond)
                
                v = spherify(z, sampling=True, sigma=max_sigma)
                x = self.decoder(v, cond, H, W)
                
                if cfg_scale > 1.0 and tags is not None:
                    x_uncond = self.decoder(v, cond_null, H, W)
                    x = x_uncond + cfg_scale * (x - x_uncond)
                    
        return x

if __name__ == "__main__":
    model = SphereAutoencoder(vocab_size=1000, patch_size=2, dim=256, enc_depth=4, dec_depth=4, latent_dim=8)
    
    # Test adaptive shapes
    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(2, 3, 32, 48)
    tags = torch.randint(0, 1000, (2, 16))
    
    recon_noisy1, recon_NOISY1, v1, cond1 = model(x1, tags)
    print("Recon1 shape:", recon_noisy1.shape)
    
    recon_noisy2, recon_NOISY2, v2, cond2 = model(x2, tags)
    print("Recon2 shape:", recon_noisy2.shape)
    
    # Test generation
    e1 = torch.randn(2, 16*16, 8)
    gen1 = model.generate(e1, 16, 16, tags, steps=2)
    print("Gen1 shape:", gen1.shape)

    e2 = torch.randn(2, 16*24, 8)
    gen2 = model.generate(e2, 16, 24, tags, steps=2)
    print("Gen2 shape:", gen2.shape)
