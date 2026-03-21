import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedder(nn.Module):
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

class GLU(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2)
        self.act = nn.SiLU()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)

class ViTBlock(nn.Module):
    """Transformer Block with 3x3 DW Conv and GLU MLP"""
    def __init__(self, dim, num_heads, r=4, num_special_tokens=7):
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # 3x3 DW Conv for positional info (applied to image tokens only)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # GLU MLP
        self.mlp = nn.Sequential(
            GLU(dim, dim * r),
            nn.Linear(dim * r, dim)
        )

    def forward(self, x):
        # x: (B, N, D)
        shortcut = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = shortcut + attn_out
        
        shortcut = x
        x = self.norm2(x)
        
        # Split special tokens and image patches
        special = x[:, :self.num_special_tokens, :]
        patches = x[:, self.num_special_tokens:, :] # (B, HW, D)
        
        # 3x3 DW Conv on patches
        B, HW, D = patches.shape
        H = W = int(math.sqrt(HW))
        patches = patches.transpose(1, 2).reshape(B, D, H, W)
        patches = self.dw_conv(patches)
        patches = patches.reshape(B, D, HW).transpose(1, 2)
        
        # Re-concat
        x = torch.cat([special, patches], dim=1)
        
        # MLP
        x = shortcut + self.mlp(x)
        return x

class EPGEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=12476):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Special Tokens: CLS (1), Time (1), Cond (1), Registers (4) = 7
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim))
        
        self.time_token_proj = nn.Linear(embed_dim, embed_dim)
        
        # Integrated Tag Processing
        self.y_embedder = nn.EmbeddingBag(num_classes + 1, embed_dim, mode='mean')
        self.y_proj = nn.Linear(embed_dim, embed_dim)
        
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, num_special_tokens=7)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.t_embedder = TimestepEmbedder(embed_dim)
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.reg_tokens, std=0.02)

    def get_y_feat(self, y_indices, y_offsets):
        """Returns projected tag features for SigLIP stage 1"""
        y_emb = self.y_embedder(y_indices, y_offsets)
        return self.y_proj(y_emb)

    def forward(self, x, t, y_indices=None, y_offsets=None):
        B = x.shape[0]
        
        # Patchify
        x = self.patch_embed(x) # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        
        # Time token
        t_emb = self.t_embedder(t)
        t_token = self.time_token_proj(t_emb).unsqueeze(1)
        
        # Cond token
        if y_indices is not None:
            y_feat = self.get_y_feat(y_indices, y_offsets)
            y_token = y_feat.unsqueeze(1)
        else:
            y_token = torch.zeros((B, 1, self.embed_dim), device=x.device)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.reg_tokens.expand(B, -1, -1)
        
        # Combine: [CLS, Time, Cond, Reg1-4, Patches]
        x = torch.cat((cls_tokens, t_token, y_token, reg_tokens, x), dim=1)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x

class EPGProjector(nn.Module):
    def __init__(self, embed_dim=768, proj_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class EPGDecoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, out_channels=3, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, num_special_tokens=7)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pred_head = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

    def forward(self, x, encoder_features=None):
        for i, block in enumerate(self.blocks):
            if encoder_features is not None:
                x = x + encoder_features[i]
            x = block(x)
            
        x = self.norm(x)
        return x

class EPGModel(nn.Module):
    def __init__(self, encoder, decoder=None, projector=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector

    def forward(self, x, t, y_indices=None, y_offsets=None, stage=1):
        if stage == 1:
            features = self.encoder(x, t, y_indices, y_offsets)
            cls_token_feat = features[:, 0]
            if self.projector:
                return self.projector(cls_token_feat), cls_token_feat
            return cls_token_feat
        else:
            # Stage 2 Fine-tuning
            enc_feats = []
            
            # Simplified forward path to capture intermediate features
            B = x.shape[0]
            x_in = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
            
            t_emb = self.encoder.t_embedder(t)
            t_token = self.encoder.time_token_proj(t_emb).unsqueeze(1)
            
            if y_indices is not None:
                y_feat = self.encoder.get_y_feat(y_indices, y_offsets)
                y_token = y_feat.unsqueeze(1)
            else:
                y_token = torch.zeros((B, 1, self.encoder.embed_dim), device=x.device)
            
            cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
            reg_tokens = self.encoder.reg_tokens.expand(B, -1, -1)
            x_in = torch.cat((cls_tokens, t_token, y_token, reg_tokens, x_in), dim=1)
            
            for block in self.encoder.blocks:
                x_in = block(x_in)
                enc_feats.append(x_in)
            
            x_enc = self.encoder.norm(x_in)
            
            # Decoder
            x_dec = x_enc
            for i, block in enumerate(self.decoder.blocks):
                x_dec = x_dec + enc_feats[-(i+1)]
                x_dec = block(x_dec)
            
            x_dec = self.decoder.norm(x_dec)
            
            # Reconstruction (Remove 7 special tokens)
            img_tokens = x_dec[:, 7:] 
            out = self.decoder.pred_head(img_tokens) 
            
            # Unpatchify
            B, N, _ = out.shape
            H = W = int(math.sqrt(N))
            P = self.decoder.patch_size
            out = out.reshape(B, H, W, P, P, -1)
            out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, H * P, W * P)
            return out

    @torch.no_grad()
    def sample_flow(self, image_size, batch_size, device, y_indices=None, y_offsets=None, steps=50):
        self.eval()
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        
        for i in range(steps):
            t_curr = ts[i]
            t_next = ts[i+1]
            t_vec = torch.full((batch_size,), t_curr, device=device)
            t_scaled = 1000 * 0.25 * torch.log(t_vec.clamp(min=1e-8))
            
            sigma_data = 0.5
            precond = 1.0 / torch.sqrt(t_curr**2 + sigma_data**2)
            x_in = x * precond
            
            x0_pred = self.forward(x_in, t_scaled, y_indices=y_indices, y_offsets=y_offsets, stage=2)
            
            if t_curr > 0:
                x = (t_next / t_curr) * x + (1 - t_next / t_curr) * x0_pred
            else:
                x = x0_pred
        return x.clamp(0, 1)
