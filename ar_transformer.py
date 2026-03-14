import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.nn.attention.flex_attention import create_block_mask

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

        omega_F = torch.cat((
            torch.zeros(n_zero_freqs),
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
        ))

        phi_hF = torch.arange(n_heads * n_freqs).reshape(n_heads, n_freqs) * direction_spacing
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)

        self.register_buffer("freqs_hF2", omega_F.unsqueeze(-1) * directions_hF2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions_BL2: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        theta = torch.einsum('hfz, blz -> bhlf', self.freqs_hF2, positions_BL2)
        cos = torch.cos(theta)
        sin = torch.sin(theta)

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


class DenoisingMLPBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        x_norm = self.norm(x)
        x_norm = x_norm * (1 + scale) + shift
        return x + gate * self.mlp(x_norm)


class DenoisingMLP(nn.Module):
    def __init__(self, dim, latent_dim, hidden_dim=1024):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.cond_proj = nn.Linear(dim * 3, hidden_dim)
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DenoisingMLPBlock(hidden_dim, hidden_dim) for _ in range(4)
        ])
        self.final_norm = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        self.final_proj = nn.Linear(hidden_dim, latent_dim)

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t * freqs[None, None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, x_t, context, tags, t):
        t_freq = self.timestep_embedding(t, 256)
        t_emb = self.time_embed(t_freq)
        c = self.cond_proj(torch.cat([t_emb, context, tags], dim=-1))
        x = self.x_proj(x_t)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_norm(x)
        return self.final_proj(x)

class SizeEmbedding(nn.Module):
    """Fourier-embeds (H, W) patch-grid dimensions into model dim."""
    def __init__(self, dim: int, fourier_dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _fourier(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) -> (B, fourier_dim)"""
        half = self.fourier_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=x.device, dtype=torch.float32)
            / half
        )
        args = x[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, H: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        h_emb = self._fourier(H.float())
        w_emb = self._fourier(W.float())
        return self.mlp(torch.cat([h_emb, w_emb], dim=-1))


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

        self.class_emb = nn.EmbeddingBag(num_classes, dim, mode='mean')
        self.patch_proj = nn.Linear(latent_dim, dim)

        # Special tokens
        self.sos_token = nn.Parameter(torch.randn(1, 1, dim))
        self.eos_token = nn.Parameter(torch.randn(1, 1, dim))  # NEW

        # Size token projection
        self.size_emb = SizeEmbedding(dim)                     # NEW

        self.rope = GGRoPE2d(num_heads, dim // num_heads)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm_f = nn.LayerNorm(dim)
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
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            if block.attn.proj.bias is not None:
                nn.init.zeros_(block.attn.proj.bias)
            nn.init.zeros_(block.mlp.w3.weight)
            if block.mlp.w3.bias is not None:
                nn.init.zeros_(block.mlp.w3.bias)
        nn.init.zeros_(self.denoise_mlp.final_proj.weight)
        nn.init.zeros_(self.denoise_mlp.final_proj.bias)

    def forward(
        self,
        patch_latents,   # (B, L, latent_dim)
        class_indices,
        positions,       # (B, L+4, 2) — Cond, Size, SOS, P1..PL, EOS
        grid_HW,         # (B, 2) int tensor — patch grid (H, W)
        offsets=None,
        block_mask=None,
        x_t=None,
        t=None,
    ):
        B, L, _ = patch_latents.shape
        device = patch_latents.device

        patch_embeddings = self.patch_proj(patch_latents)  # (B, L, dim)
        global_tags = self.class_emb(class_indices, offsets)

        if B == 1 and global_tags.size(0) > 1:
            global_tags = global_tags[0:1]

        # Special tokens
        size_embed = self.size_emb(grid_HW[:, 0], grid_HW[:, 1])  # (B, dim)
        cond_token = global_tags.unsqueeze(1)                       # (B, 1, dim)
        size_token = size_embed.unsqueeze(1)                        # (B, 1, dim)
        sos = self.sos_token.expand(B, -1, -1)                      # (B, 1, dim)
        eos = self.eos_token.expand(B, -1, -1)                      # (B, 1, dim)

        # Input:  [Cond, Size, SOS, P1, ..., PL-1, EOS]
        # Target: [P1,   P2,  P3,  ..., PL,  EOS-pred  ] (shifted by 1)
        # We feed EOS at the end so the model learns to predict it after last patch
        x = torch.cat([cond_token, size_token, sos, patch_embeddings[:, :-1], eos], dim=1)

        pos_seq = positions[:, :x.size(1)]

        for block in self.blocks:
            x = block(x, self.rope, pos_seq, block_mask)

        x = self.norm_f(x)

        # Drop Cond output, keep Size through EOS outputs for prediction
        # x[:, 0] = Cond output  -> discard
        # x[:, 1:] = Size, SOS, P1..PL-1, EOS outputs -> predict P1..PL + EOS
        x_for_pred = x[:, 1:]  # (B, L+2, dim)  Size/SOS/patches/EOS hidden states
        seq_len = x_for_pred.size(1)
        if x_t is None:
            x_t = torch.randn(B, seq_len, self.latent_dim, device=device)
        if t is None:
            t = torch.ones(B, seq_len, 1, device=device)

        tags_expanded = global_tags.unsqueeze(1).expand(-1, seq_len, -1)
        pred_x = self.denoise_mlp(x_t, x_for_pred, tags_expanded, t)
        return pred_x  # (B, L+3, latent_dim)

    @torch.no_grad()
    def generate(
            self,
            class_indices,
            grid_H: int,
            grid_W: int,
            device='cuda',
    ):
        B = 1 if class_indices.dim() == 1 else class_indices.size(0)

        if class_indices.dim() == 1:
            offsets = torch.tensor([0], device=device)
        else:
            B, N = class_indices.shape
            offsets = torch.arange(0, B * N, N, device=device)
            class_indices = class_indices.flatten()

        global_tags = self.class_emb(class_indices, offsets)

        grid_HW = torch.tensor([[grid_H, grid_W]], device=device).expand(B, -1)
        size_embed = self.size_emb(grid_HW[:, 0].float(), grid_HW[:, 1].float())

        cond  = global_tags.unsqueeze(1)       # (B, 1, dim)
        size  = size_embed.unsqueeze(1)        # (B, 1, dim)
        sos   = self.sos_token.expand(B, -1, -1)

        # Sequence starts as [Cond, Size, SOS]
        current_seq_emb = torch.cat([cond, size, sos], dim=1)

        # Coordinate scheme matching get_2d_positions() in training
        xlim = math.sqrt(grid_W / grid_H)
        ylim = math.sqrt(grid_H / grid_W)
        xs = torch.linspace(-xlim, xlim, grid_W, device=device)
        ys = torch.linspace(-ylim, ylim, grid_H, device=device)

        # Position indices: 0=Cond, 1=Size, 2=SOS, 3..=patches, last=EOS
        def get_pos(idx, total_patches):
            if idx in (0, 1, 2):
                return [0.0, 0.0]
            patch_idx = idx - 3
            if patch_idx >= total_patches:  # EOS position
                return [0.0, 0.0]
            r, c = patch_idx // grid_W, patch_idx % grid_W
            return [xs[c].item(), ys[r].item()]

        generated_latents = []
        max_patches = grid_H * grid_W

        for i in range(max_patches):
            seq_len = current_seq_emb.size(1)
            positions = torch.tensor(
                [get_pos(j, max_patches) for j in range(seq_len)],
                device=device
            ).unsqueeze(0).expand(B, -1, -1)

            x = current_seq_emb
            for block in self.blocks:
                x = block(x, self.rope, positions)

            x_last = self.norm_f(x[:, -1:])
            x_t = torch.randn(B, 1, self.latent_dim, device=device)
            tags_expanded = global_tags.unsqueeze(1)

            steps = 50
            for step_idx in range(steps):
                t_val = step_idx / steps
                t = torch.full((B, 1, 1), t_val, device=device)
                v_pred = self.denoise_mlp(x_t, x_last, tags_expanded, t)
                if t_val < 0.999:
                    v = (v_pred - x_t) / (1.0 - t_val)
                    x_t = x_t + v * (1.0 / steps)
                else:
                    x_t = v_pred

            next_latent = torch.where(x_t > 0.5, 1.0, torch.where(x_t < -0.5, -1.0, 0.0))
            generated_latents.append(next_latent)

            next_emb = self.patch_proj(next_latent)
            current_seq_emb = torch.cat([current_seq_emb, next_emb], dim=1)

            if current_seq_emb.size(1) > self.max_seq_len:
                current_seq_emb = torch.cat(
                    [current_seq_emb[:, :2], current_seq_emb[:, -(self.max_seq_len - 2):]],
                    dim=1
                )

        return torch.cat(generated_latents, dim=1)