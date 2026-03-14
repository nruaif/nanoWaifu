import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Check for FlexAttention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions_BL2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k:            (B, h, L, d)
        positions_BL2:   (B, L, 2)
        """
        theta = torch.einsum('hfz, blz -> bhlf', self.freqs_hF2, positions_BL2)
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        def rotate_apply(x):
            x1, x2 = x.float().chunk(2, dim=-1)
            out1 = x1 * cos - x2 * sin
            out2 = x1 * sin + x2 * cos
            return torch.cat((out1, out2), dim=-1).type_as(x)

        return rotate_apply(q), rotate_apply(k)


# ---------------------------------------------------------------------------
# Attention + MLP blocks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Denoising MLP
# ---------------------------------------------------------------------------

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
        x_norm = self.norm(x) * (1 + scale) + shift
        return x + gate * self.mlp(x_norm)


class DenoisingMLP(nn.Module):
    def __init__(self, dim, latent_dim, hidden_dim=1024):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        # context=dim, tags=dim, t_emb=dim  ->  hidden_dim
        self.cond_proj = nn.Linear(dim * 3, hidden_dim)
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DenoisingMLPBlock(hidden_dim, hidden_dim) for _ in range(4)
        ])
        self.final_norm = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        self.final_proj = nn.Linear(hidden_dim, latent_dim)

    def timestep_embedding(self, t, dim, max_period=10000):
        # t: (B, L, 1)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t * freqs[None, None, :]          # (B, L, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, L, dim)

    def forward(self, x_t, context, tags, t):
        """
        x_t:     (B, L, latent_dim)
        context: (B, L, dim)        transformer hidden states
        tags:    (B, L, dim)        global tag embedding expanded
        t:       (B, L, 1)          diffusion timestep in [0, 1]
        """
        t_emb = self.time_embed(self.timestep_embedding(t, 256))  # (B, L, dim)
        c = self.cond_proj(torch.cat([t_emb, context, tags], dim=-1))  # (B, L, hidden_dim)
        x = self.x_proj(x_t)
        for block in self.blocks:
            x = block(x, c)
        return self.final_proj(self.final_norm(x))


# ---------------------------------------------------------------------------
# Size token embedding
# ---------------------------------------------------------------------------

class SizeEmbedding(nn.Module):
    """
    Fourier-embeds the patch-grid dimensions (H, W) and projects to model dim.
    Used as a dedicated Size token in the sequence.
    """
    def __init__(self, dim: int, fourier_dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.max_period = max_period
        # _fourier(x) -> (B, fourier_dim);  cat([h, w]) -> (B, fourier_dim*2)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # zero-init so it starts as a no-op and learns gradually
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _fourier(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) float -> (B, fourier_dim)"""
        half = self.fourier_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=x.device, dtype=torch.float32)
            / half
        )
        args = x[:, None] * freqs[None]                               # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, fourier_dim)

    def forward(self, H: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """H, W: (B,) int tensors  ->  (B, dim)"""
        return self.mlp(torch.cat([self._fourier(H.float()), self._fourier(W.float())], dim=-1))


# ---------------------------------------------------------------------------
# AR Transformer
# ---------------------------------------------------------------------------

# Sequence layout
# ┌─────────────────────────────────────────────────────────────┐
# │ Training input:   [Cond, Size, SOS, P1,  P2,  … PL-1, EOS] │
# │ Training target:  [P1,   P2,  P3,  P4,  P5,  … PL,   ---] │
# │                                                             │
# │ Special token positions (all at [0,0]):                     │
# │   idx 0 = Cond                                              │
# │   idx 1 = Size                                              │
# │   idx 2 = SOS                                               │
# │   idx 3..L+2 = patches P1..PL                               │
# │   idx L+3 = EOS                                             │
# └─────────────────────────────────────────────────────────────┘
#
# forward() returns pred_x of shape (B, L, latent_dim):
#   pred_x[:, 0]   <- predicted from Size  hidden state  (target P1)
#   pred_x[:, 1]   <- predicted from SOS   hidden state  (target P2)
#   pred_x[:, i]   <- predicted from Pi-1  hidden state  (target Pi+1)
#   pred_x[:, L-1] <- predicted from PL-1  hidden state  (target PL)
#
# The EOS hidden state is computed but not returned — no continuous target.

class ARTransformer(nn.Module):
    # Number of non-patch tokens prepended/appended to the sequence
    N_PREFIX = 3   # Cond, Size, SOS
    N_SUFFIX = 1   # EOS

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
        self.dim = dim

        self.class_emb = nn.EmbeddingBag(num_classes, dim, mode='mean')
        self.patch_proj = nn.Linear(latent_dim, dim)

        # Special tokens (learned parameters)
        self.sos_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.eos_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Size token: fourier embed of (H, W) patch grid
        self.size_emb = SizeEmbedding(dim)

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
        patch_latents,  # (B, L, latent_dim)  — float, values in {-1,0,1}
        class_indices,  # (total_tags,)        — for EmbeddingBag
        positions,      # (B, L+4, 2)          — from get_2d_positions()
        grid_HW,        # (B, 2)               — patch grid height & width per sample
        offsets=None,   # (B,)                 — EmbeddingBag offsets
        block_mask=None,
        x_t=None,       # (B, L, latent_dim)   — noised latents; sampled if None
        t=None,         # (B, L, 1)            — diffusion time in [0,1]; ones if None
    ):
        B, L, _ = patch_latents.shape
        device = patch_latents.device

        # --- Embeddings ---
        patch_emb   = self.patch_proj(patch_latents)               # (B, L,   dim)
        global_tags = self.class_emb(class_indices, offsets)        # (B,      dim)
        if B == 1 and global_tags.size(0) > 1:
            global_tags = global_tags[0:1]

        cond_tok = global_tags.unsqueeze(1)                         # (B, 1,   dim)
        size_tok = self.size_emb(grid_HW[:, 0], grid_HW[:, 1]).unsqueeze(1)  # (B, 1, dim)
        sos_tok  = self.sos_token.expand(B, -1, -1)                 # (B, 1,   dim)
        eos_tok  = self.eos_token.expand(B, -1, -1)                 # (B, 1,   dim)

        # Input sequence:  [Cond, Size, SOS, P1, P2, … PL-1, EOS]   length = L+3
        # Predicts target: [P1,   P2,  P3,  P4, P5, … PL,   EOS*]
        # (*EOS target not used — we slice it off before the denoising MLP)
        x = torch.cat([cond_tok, size_tok, sos_tok, patch_emb[:, :-1], eos_tok], dim=1)
        # x shape: (B, L+3, dim)

        pos_seq = positions[:, :x.size(1)]                          # (B, L+3, 2)

        for block in self.blocks:
            x = block(x, self.rope, pos_seq, block_mask)

        x = self.norm_f(x)                                          # (B, L+3, dim)

        # Drop Cond output (idx 0), keep Size/SOS/patch outputs (idx 1..L),
        # drop EOS output (idx L+1 and L+2) — no continuous latent target for it.
        # This gives exactly L hidden states aligned with L patch targets.
        #
        #   x[:, 0]   = Cond  -> discarded
        #   x[:, 1]   = Size  -> predicts P1   (x_for_pred[:, 0])
        #   x[:, 2]   = SOS   -> predicts P2   (x_for_pred[:, 1])
        #   x[:, 3]   = P1    -> predicts P3   (x_for_pred[:, 2])
        #   ...
        #   x[:, L]   = PL-1  -> predicts PL   (x_for_pred[:, L-1])
        #   x[:, L+1] = EOS   -> discarded
        #   x[:, L+2] = (pad) -> discarded
        x_for_pred = x[:, 1:L+1]                                    # (B, L, dim)

        # Denoising MLP
        if x_t is None:
            x_t = torch.randn(B, L, self.latent_dim, device=device)
        if t is None:
            t = torch.ones(B, L, 1, device=device)

        tags_exp = global_tags.unsqueeze(1).expand(-1, L, -1)       # (B, L, dim)
        pred_x   = self.denoise_mlp(x_t, x_for_pred, tags_exp, t)   # (B, L, latent_dim)
        return pred_x

    # -----------------------------------------------------------------------
    # Autoregressive generation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate(
            self,
            class_indices,  # (N,) or (B, N)
            grid_H: int,    # patch grid height
            grid_W: int,    # patch grid width
            steps: int = 50,
            device: str = 'cuda',
    ):
        if class_indices.dim() == 1:
            B = 1
            offsets = torch.tensor([0], device=device)
        else:
            B, N = class_indices.shape
            offsets = torch.arange(0, B * N, N, device=device)
            class_indices = class_indices.flatten()

        global_tags = self.class_emb(class_indices, offsets)        # (B, dim)

        # Build starting sequence: [Cond, Size, SOS]
        cond_tok = global_tags.unsqueeze(1)                          # (B, 1, dim)
        size_tok = self.size_emb(
            torch.full((B,), grid_H, device=device, dtype=torch.float32),
            torch.full((B,), grid_W, device=device, dtype=torch.float32),
        ).unsqueeze(1)                                               # (B, 1, dim)
        sos_tok  = self.sos_token.expand(B, -1, -1)                  # (B, 1, dim)

        current_seq = torch.cat([cond_tok, size_tok, sos_tok], dim=1)  # (B, 3, dim)

        # Coordinate grid — matches get_2d_positions() in train.py exactly
        xlim = math.sqrt(grid_W / grid_H)
        ylim = math.sqrt(grid_H / grid_W)
        xs = torch.linspace(-xlim, xlim, grid_W, device=device)
        ys = torch.linspace(-ylim, ylim, grid_H, device=device)

        def get_pos(idx: int) -> list:
            # 0=Cond, 1=Size, 2=SOS  -> [0,0]
            # 3..3+H*W-1             -> patch coords
            if idx < self.N_PREFIX:
                return [0.0, 0.0]
            patch_idx = idx - self.N_PREFIX
            r, c = patch_idx // grid_W, patch_idx % grid_W
            return [xs[c].item(), ys[r].item()]

        generated = []
        max_patches = grid_H * grid_W
        tags_exp = global_tags.unsqueeze(1)                          # (B, 1, dim)

        for i in range(max_patches):
            seq_len = current_seq.size(1)
            pos = torch.tensor(
                [get_pos(j) for j in range(seq_len)], device=device
            ).unsqueeze(0).expand(B, -1, -1)                        # (B, seq_len, 2)

            # Run transformer, take last hidden state
            x = current_seq
            for block in self.blocks:
                x = block(x, self.rope, pos)
            x_last = self.norm_f(x[:, -1:])                         # (B, 1, dim)

            # Flow matching denoising: noise -> latent
            x_t = torch.randn(B, 1, self.latent_dim, device=device)
            for step_idx in range(steps):
                t_val = step_idx / steps
                t = torch.full((B, 1, 1), t_val, device=device)
                v_pred = self.denoise_mlp(x_t, x_last, tags_exp, t)
                if t_val < (1.0 - 1.0 / steps):
                    x_t = x_t + (v_pred - x_t) / (1.0 - t_val) * (1.0 / steps)
                else:
                    x_t = v_pred

            # Ternary quantization {-1, 0, 1}
            next_latent = torch.where(x_t > 0.5, 1.0, torch.where(x_t < -0.5, -1.0, 0.0))
            generated.append(next_latent)

            # Append embedding to sequence
            next_emb = self.patch_proj(next_latent)                  # (B, 1, dim)
            current_seq = torch.cat([current_seq, next_emb], dim=1)

            # Truncate if needed, always keep Cond+Size+SOS prefix intact
            if current_seq.size(1) > self.max_seq_len:
                prefix = current_seq[:, :self.N_PREFIX]
                rest   = current_seq[:, -(self.max_seq_len - self.N_PREFIX):]
                current_seq = torch.cat([prefix, rest], dim=1)

        return torch.cat(generated, dim=1)                           # (B, H*W, latent_dim)