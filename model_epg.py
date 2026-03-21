import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
import os
import random

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('weight', None)

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * norm
        if self.elementwise_affine:
            x = x * self.weight
        return x

class QKNormAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # QK Norm
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TagProcessor:
    def __init__(self, tags_file, characters_file="characters.csv"):
        with open(tags_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.num_classes = len(self.tags)
        
        # Character labels for Supervised Contrastive Learning (SupCon)
        if os.path.exists(characters_file):
            import pandas as pd
            # Using latin-1 encoding to match dataset.py and avoid UnicodeDecodeErrors
            df = pd.read_csv(characters_file, encoding='latin-1')
            self.character_set = set(df['character'].astype(str).tolist())
            self.char_to_id = {str(c): i for i, c in enumerate(df['character'].tolist())}
        else:
            self.character_set = set()
            self.char_to_id = {}

    def process_prompts(self, prompts, device, dropout_prob=0.0):
        indices = []
        offsets = [0]
        labels = []
        for p in prompts:
            tags = p.split() if isinstance(p, str) else []
            
            # Extract tags for embedding
            if random.random() < dropout_prob:
                indices.append(self.num_classes)
            else:
                count = 0
                for t in tags:
                    if t in self.tag_to_idx:
                        indices.append(self.tag_to_idx[t])
                        count += 1
                if count == 0:
                    indices.append(self.num_classes)
            offsets.append(len(indices))
            
            # Extract label for SupCon (supervised part)
            found_char = False
            for t in tags:
                if t in self.character_set:
                    labels.append(self.char_to_id[t])
                    found_char = True
                    break
            if not found_char:
                labels.append(-1)

        indices = torch.tensor(indices, dtype=torch.long, device=device)
        offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        return indices, offsets, labels

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
    def __init__(self, dim, num_heads, r=4, num_special_tokens=7):
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = QKNormAttention(dim, num_heads)
        
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.mlp = nn.Sequential(GLU(dim, dim * r), nn.Linear(dim * r, dim))

    def forward(self, x, h, w):
        shortcut = x
        x = self.norm1(x)
        x = shortcut + self.attn(x)
        
        shortcut = x
        x = self.norm2(x)
        
        special = x[:, :self.num_special_tokens, :]
        patches = x[:, self.num_special_tokens:, :]
        B, HW, D = patches.shape
        patches = patches.transpose(1, 2).reshape(B, D, h, w)
        patches = self.dw_conv(patches)
        patches = patches.reshape(B, D, HW).transpose(1, 2)
        
        x = torch.cat([special, patches], dim=1)
        x = shortcut + self.mlp(x)
        return x

class EPGEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=12476):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.time_token_proj = nn.Linear(embed_dim, embed_dim)
        self.y_embedder = nn.EmbeddingBag(num_classes + 1, embed_dim, mode='mean')
        self.y_proj = nn.Linear(embed_dim, embed_dim)
        
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads, num_special_tokens=7) for _ in range(depth)])
        self.norm = RMSNorm(embed_dim, elementwise_affine=False)
        self.t_embedder = TimestepEmbedder(embed_dim)
        
        self.gradient_checkpointing = False
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.reg_tokens, std=0.02)

    def get_y_feat(self, y_indices, y_offsets):
        y_emb = self.y_embedder(y_indices, y_offsets)
        return self.y_proj(y_emb)

    def forward(self, x, t, y_indices=None, y_offsets=None):
        B, C, H_img, W_img = x.shape
        h, w = H_img // self.patch_size, W_img // self.patch_size
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        t_emb = self.t_embedder(t)
        t_token = self.time_token_proj(t_emb).unsqueeze(1)
        
        if y_indices is not None:
            y_token = self.get_y_feat(y_indices, y_offsets).unsqueeze(1)
        else:
            y_token = torch.zeros((B, 1, self.embed_dim), device=x.device)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.reg_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, t_token, y_token, reg_tokens, x), dim=1)
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, h, w, use_reentrant=False)
            else:
                x = block(x, h, w)
        return self.norm(x)

class EPGProjector(nn.Module):
    def __init__(self, embed_dim=768, proj_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            RMSNorm(embed_dim, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            RMSNorm(embed_dim, elementwise_affine=False),
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
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads, num_special_tokens=7) for _ in range(depth)])
        self.norm = RMSNorm(embed_dim, elementwise_affine=False)
        self.pred_head = nn.Linear(embed_dim, out_channels * patch_size * patch_size)
        
        self.gradient_checkpointing = False

    def forward(self, x, h, w, encoder_features=None):
        for i, block in enumerate(self.blocks):
            if encoder_features is not None: x = x + encoder_features[i]
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, h, w, use_reentrant=False)
            else:
                x = block(x, h, w)
        return self.norm(x)

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
            if self.projector: return self.projector(cls_token_feat), cls_token_feat
            return cls_token_feat
        else:
            B, C, H_img, W_img = x.shape
            h, w = H_img // self.encoder.patch_size, W_img // self.encoder.patch_size
            enc_feats = []
            x_in = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
            t_token = self.encoder.time_token_proj(self.encoder.t_embedder(t)).unsqueeze(1)
            y_token = self.encoder.get_y_feat(y_indices, y_offsets).unsqueeze(1) if y_indices is not None else torch.zeros((B, 1, self.encoder.embed_dim), device=x.device)
            x_in = torch.cat((self.encoder.cls_token.expand(B, -1, -1), t_token, y_token, self.encoder.reg_tokens.expand(B, -1, -1), x_in), dim=1)
            for block in self.encoder.blocks:
                x_in = block(x_in, h, w)
                enc_feats.append(x_in)
            x_dec = self.encoder.norm(x_in)
            for i, block in enumerate(self.decoder.blocks):
                x_dec = x_dec + enc_feats[-(i+1)]
                x_dec = block(x_dec, h, w)
            x_dec = self.decoder.norm(x_dec)
            out = self.decoder.pred_head(x_dec[:, 7:])
            out = out.reshape(B, h, w, self.decoder.patch_size, self.decoder.patch_size, -1)
            return out.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, h * self.decoder.patch_size, w * self.decoder.patch_size)

    @torch.no_grad()
    def sample_flow(self, image_size, batch_size, device, y_indices=None, y_offsets=None, steps=50):
        self.eval()
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t_curr, t_next = ts[i], ts[i+1]
            t_scaled = 1000 * 0.25 * torch.log(torch.full((batch_size,), t_curr, device=device).clamp(min=1e-8))
            x_in = x * (1.0 / torch.sqrt(t_curr**2 + 0.5**2))
            x0_pred = self.forward(x_in, t_scaled, y_indices=y_indices, y_offsets=y_offsets, stage=2)
            x = (t_next / t_curr) * x + (1 - t_next / t_curr) * x0_pred if t_curr > 0 else x0_pred
        return x.clamp(0, 1)
