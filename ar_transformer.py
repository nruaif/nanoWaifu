import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Check for FlexAttention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class GGRoPE2d(nn.Module):
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        min_freq: float = 0.5,
        max_freq: float = 10.0,
        p_zero_freqs: float = 0.0,
        direction_spacing: float = math.pi * (math.sqrt(5) - 1) / 2,
    ):
        super().__init__()
        assert head_dim % 2 == 0
        n_freqs = head_dim // 2
        n_zero_freqs = round(p_zero_freqs * n_freqs)
        
        # Frequency magnitudes
        omega_F = torch.cat((
            torch.zeros(n_zero_freqs),
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
        ))
        
        # Directions
        phi_hF = torch.arange(n_heads * n_freqs).reshape(n_heads, n_freqs) * direction_spacing
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
        
        # Store freqs: (n_heads, n_freqs, 2)
        self.register_buffer("freqs_hF2", omega_F.unsqueeze(-1) * directions_hF2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions_BL2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (B, h, L, d)
        positions_BL2: (B, L, 2) - normalized coordinates
        """
        # theta = (freqs_hF2[h, F, 2] * positions_BL2[B, L, 2]).sum(-1) -> (B, h, L, F)
        theta = torch.einsum('hfz, blz -> bhlf', self.freqs_hF2, positions_BL2)
        
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        # Apply RoPE: (B, h, L, F)
        def rotate_apply(x):
            x1, x2 = x.float().chunk(2, dim=-1)
            out1 = x1 * cos - x2 * sin
            out2 = x1 * sin + x2 * cos
            return torch.cat((out1, out2), dim=-1).type_as(x)
            
        return rotate_apply(q), rotate_apply(k)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, rope: GGRoPE2d, positions: torch.Tensor, block_mask=None):
        B, L, C = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        # Apply 2D GGRoPE
        q, k = rope(q, k, positions)

        if HAS_FLEX:
            def score_mod(score, b, h, q_idx, kv_idx):
                return 15.0 * torch.tanh(score / 15.0)
            y = flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)
        else:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = 15.0 * torch.tanh(attn / 15.0)
            
            if block_mask is not None:
                attn = attn + block_mask
            else:
                mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
                attn = attn.masked_fill(mask, -float('inf'))
            
            attn = F.softmax(attn, dim=-1)
            y = attn @ v
            
        y = y.transpose(1, 2).reshape(B, L, C)
        return self.proj(y)

class GLU_ReLU2_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # GLU with ReLU^2
        # (w1(x) * ReLU(w2(x))^2) * w3
        x = self.w1(x) * torch.relu(self.w2(x)).pow(2)
        x = self.w3(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GLU_ReLU2_MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, rope=None, positions=None, block_mask=None):
        x = x + self.attn(self.norm1(x), rope, positions, block_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class DenoisingMLP(nn.Module):
    def __init__(self, dim, latent_dim, hidden_dim=1024):
        super().__init__()
        # Input: noise(latent_dim) + cond(dim) + tags(dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, noise, cond, tags):
        # noise: (B, L, latent_dim)
        # cond: (B, L, dim)
        # tags: (B, L, dim)
        x = torch.cat([noise, cond, tags], dim=-1)
        return self.mlp(x)

class ARTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=256,
        dim=512,
        depth=12,
        num_heads=8,
        max_seq_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        
        # Class Embedding (Global tags)
        self.class_emb = nn.EmbeddingBag(num_classes, dim, mode='mean')
        
        # Patch Projection (Input bit latents are mapped to transformer dim)
        self.patch_proj = nn.Linear(latent_dim, dim)
        
        # Special tokens
        self.sos_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 2D GGRoPE
        self.rope = GGRoPE2d(num_heads, dim // num_heads)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm_f = nn.LayerNorm(dim)
        
        # Denoising MLP (4 layers total inside Sequential)
        self.denoise_mlp = DenoisingMLP(dim, latent_dim)
        
        self.apply(self._init_weights)
        self.zero_init_outputs()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def zero_init_outputs(self):
        """Zero initialize output layers of attention and MLP for stability."""
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            if block.attn.proj.bias is not None:
                nn.init.zeros_(block.attn.proj.bias)
            
            nn.init.zeros_(block.mlp.w3.weight)
            if block.mlp.w3.bias is not None:
                nn.init.zeros_(block.mlp.w3.bias)
        
        # Final layer of denoising MLP
        nn.init.zeros_(self.denoise_mlp.mlp[-1].weight)
        nn.init.zeros_(self.denoise_mlp.mlp[-1].bias)

    def forward(self, patch_latents, class_indices, positions, offsets=None, block_mask=None, noise=None):
        """
        patch_latents: (B, L, 256) - ternary values {-1, 0, 1}
        positions: (B, L+2, 2) - Includes Cond and SOS coordinates
        """
        B, L, _ = patch_latents.shape
        device = patch_latents.device
        
        patch_embeddings = self.patch_proj(patch_latents)
        global_tags = self.class_emb(class_indices, offsets)
        
        # If packing multiple samples into B=1, handle global_tags mismatch
        if B == 1 and global_tags.size(0) > 1:
            # For now, take only the first one to match sequence start
            # A more advanced version would use image-specific tags for each position
            global_tags = global_tags[0:1]
            
        cond_embeddings = global_tags.unsqueeze(1)
        sos = self.sos_token.expand(B, -1, -1)
        
        # Input: [Cond, SOS, P1, ..., PL-1]
        x = torch.cat([cond_embeddings, sos, patch_embeddings[:, :-1]], dim=1)
        
        # Ensure positions match sequence length
        pos_seq = positions[:, :x.size(1)]
        
        for block in self.blocks:
            x = block(x, self.rope, pos_seq, block_mask)
            
        x = self.norm_f(x)
        
        if noise is None:
            noise = torch.randn(B, x.size(1), self.latent_dim, device=device)
            
        # Condition MLP on transformer hidden states and global tags
        tags_expanded = global_tags.unsqueeze(1).expand(-1, x.size(1), -1)
        pred_x = self.denoise_mlp(noise, x, tags_expanded)
            
        return pred_x

    @torch.no_grad()
    def generate(
        self, 
        class_indices, 
        max_patches=256, 
        device='cuda'
    ):
        B = class_indices.size(0) if hasattr(class_indices, 'size') and class_indices.dim() > 1 else 1
        
        if class_indices.dim() == 1:
            offsets = torch.tensor([0], device=device)
        else:
            B, N = class_indices.shape
            offsets = torch.arange(0, B * N, N, device=device)
            class_indices = class_indices.flatten()
            
        global_tags = self.class_emb(class_indices, offsets)
        cond = global_tags.unsqueeze(1)
        sos = self.sos_token.expand(B, -1, -1)
        
        current_seq_emb = torch.cat([cond, sos], dim=1)
        
        grid_size = int(math.sqrt(max_patches))
        
        def get_pos(idx, grid_size):
            if idx == 0: return [0.0, 0.0] # Cond
            if idx == 1: return [0.0, 0.0] # SOS
            patch_idx = idx - 2
            r, c = patch_idx // grid_size, patch_idx % grid_size
            return [2 * (c / (grid_size - 1)) - 1, 2 * (r / (grid_size - 1)) - 1]

        generated_latents = []
        
        for i in range(max_patches):
            seq_len = current_seq_emb.size(1)
            positions = torch.tensor([get_pos(j, grid_size) for j in range(seq_len)], device=device).unsqueeze(0).expand(B, -1, -1)
            
            x = current_seq_emb
            for block in self.blocks:
                x = block(x, self.rope, positions)
            
            x_last = self.norm_f(x[:, -1:])
            
            noise = torch.randn(B, 1, self.latent_dim, device=device)
            tags_expanded = global_tags.unsqueeze(1)
            
            pred_x = self.denoise_mlp(noise, x_last, tags_expanded)
            
            # Binary/Ternary quantization for the next step input
            # Using simple sign for now, or could keep it continuous if desired.
            # "bit latent" suggests we want it to be discrete.
            next_latent = torch.where(pred_x > 0.5, 1.0, torch.where(pred_x < -0.5, -1.0, 0.0))
            
            generated_latents.append(next_latent)
            
            next_emb = self.patch_proj(next_latent)
            current_seq_emb = torch.cat([current_seq_emb, next_emb], dim=1)
            
            if current_seq_emb.size(1) > self.max_seq_len:
                current_seq_emb = torch.cat([current_seq_emb[:, :1], current_seq_emb[:, -(self.max_seq_len-1):]], dim=1)
                
        return torch.cat(generated_latents, dim=1)

