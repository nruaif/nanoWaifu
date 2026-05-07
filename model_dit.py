import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import lru_cache
from typing import Tuple
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts,
    CheckpointPolicy,
)
import functools

aten = torch.ops.aten

compute_intensive_ops = [
    aten.mm.default,
    aten.bmm.default,
    aten.addmm.default,
]


def policy_fn(ctx, op, *args, **kwargs):
    if op in compute_intensive_ops:
        return CheckpointPolicy.MUST_SAVE
    else:
        return CheckpointPolicy.PREFER_RECOMPUTE


_ = torch._dynamo

context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

torch._functorch.config.activation_memory_budget = 0.5


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0):
    y_pos = torch.arange(height, dtype=torch.float32)
    x_pos = torch.arange(width, dtype=torch.float32)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), eps=self.eps)


Norm = RMSNorm


class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.constant_(self.w2.weight, 0)
        self.act = ReLUSquared()

    def forward(self, x):
        return self.w2((F.relu(self.w1(x)) ** 2))


FeedForward = SwiGLU


class FeedForwardDW(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        nn.init.constant_(self.w2.weight, 0)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x1 = self.w1(x)
        x1_spatial = x1.transpose(1, 2).reshape(B, -1, H, W)
        x1_spatial = self.dwconv(x1_spatial)
        x1 = x1_spatial.flatten(2).transpose(1, 2)

        return self.w2((F.relu(x1) ** 2))


class Embed(nn.Module):
    def __init__(self, in_features, out_features, bias=False, norm_layer=None):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)
        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            ReLUSquared(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(dtype=t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.,
                 proj_drop: float = 0.) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.constant_(self.proj.weight, 0)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, pos, num_txt_tokens: int, v_skip: torch.Tensor = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())

        q_txt, q_img = q[:, :, :num_txt_tokens], q[:, :, num_txt_tokens:]
        k_txt, k_img = k[:, :, :num_txt_tokens], k[:, :, num_txt_tokens:]

        # q_img, k_img = apply_rotary_emb(q_img, k_img, freqs_cis=pos)

        q = torch.cat([q_txt, q_img], dim=2)
        k = torch.cat([k_txt, k_img], dim=2)

        v_out = v
        if v_skip is not None:
            v = v * self.alpha + v_skip * self.beta

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, v_out


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2_txt = Norm(hidden_size, eps=1e-6)
        self.norm2_img = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_txt = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_img = FeedForwardDW(hidden_size, mlp_hidden_dim)

    @torch.compile
    def forward(self, x, pos, num_txt_tokens, H, W, v_skip=None):
        attn_out, v_out = self.attn(self.norm1(x), pos, num_txt_tokens, v_skip=v_skip)
        x = x + attn_out

        x_txt, x_img = x[:, :num_txt_tokens], x[:, num_txt_tokens:]

        x_txt = x_txt + self.mlp_txt(self.norm2_txt(x_txt))
        x_img = x_img + self.mlp_img(self.norm2_img(x_img), H, W)

        x = torch.cat([x_txt, x_img], dim=1)
        return x, v_out


class PixNerDiT(nn.Module):
    def __init__(
            self,
            in_channels=3,
            num_groups=16,
            hidden_size=1024,
            num_encoder_blocks=16,
            patch_size=16,
            vocab_size=1024,
            txt_max_length=32,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_encoder_blocks = num_encoder_blocks
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.txt_max_length = txt_max_length
        self.s_embedder = Embed(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.y_embedder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            Norm(hidden_size)
        )

        self.y_pos_embedding = torch.nn.Parameter(
            torch.randn(1, txt_max_length, hidden_size),
            requires_grad=True
        )

        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_encoder_blocks)
        ])

        self.final_conv = nn.Conv2d(self.hidden_size, self.in_channels * self.patch_size ** 2, kernel_size=3, padding=1)

        self.initialize_weights()
        self.precompute_pos = dict()
        self.weight_path = weight_path
        self.load_ema = load_ema
        self.grad_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.grad_checkpointing = True

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_conv.weight, 0)

    def forward(self, x, t, y):
        B, _, H, W = x.shape
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        x_unfolded = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        xpos = self.fetch_pos(H_patch, W_patch, x.device)
        ypos = self.y_pos_embedding[:, :y.shape[1], :]
        t_emb = self.t_embedder(t.view(-1)).view(B, 1, self.hidden_size)

        y_emb = self.y_embedder(y).view(B, -1, self.hidden_size) + ypos.to(x.dtype)

        s = self.s_embedder(x_unfolded)
        s = s + t_emb.view(B, 1, self.hidden_size)

        num_txt_tokens = y_emb.shape[1]
        seq = torch.cat([y_emb, s], dim=1)

        v_skips = []
        num_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            if i < num_blocks // 2:
                v_skip = None
            elif i > num_blocks // 2 or (num_blocks % 2 == 0 and i >= num_blocks // 2):
                v_skip = v_skips.pop()
            else:
                v_skip = None

            if self.grad_checkpointing and self.training:
                seq, v_out = checkpoint(block, seq, xpos, num_txt_tokens, H_patch, W_patch, v_skip, use_reentrant=False,
                                        context_fn=context_fn)
            else:
                seq, v_out = block(seq, xpos, num_txt_tokens, H_patch, W_patch, v_skip)

            if i < num_blocks // 2:
                v_skips.append(v_out)

        s = seq[:, num_txt_tokens:]

        s = s.transpose(1, 2).reshape(B, self.hidden_size, H_patch, W_patch)
        x_out = self.final_conv(s)
        x_out = x_out.reshape(B, self.in_channels * self.patch_size ** 2, -1)
        x_out = torch.nn.functional.fold(x_out, (H, W), kernel_size=self.patch_size, stride=self.patch_size)

        return x_out

    @torch.no_grad()
    def sample(self, B, H, W, device, steps=50, y=None, cfg_scale=4.0, pad_idx=None):
        x = torch.randn(B, self.in_channels, H, W, device=device)
        dt = 1.0 / steps
        for step in range(steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device) * 1000

            if cfg_scale > 1.0 and y is not None and pad_idx is not None:
                x_in = x.repeat(2, 1, 1, 1)
                t_in = t.repeat(2)
                y_uncond = y.clone()
                y_uncond[:] = pad_idx
                y_in = torch.cat([y, y_uncond], dim=0)

                out = self(x_in, t_in, y_in)
                out_cond, out_uncond = out.chunk(2, dim=0)
                x0_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
            else:
                x0_pred = self(x, t, y)

            denom = max(1.0 - t_val, 0.05)
            v_pred = (x0_pred - x) / denom
            x = x + v_pred * dt

        return x


def print_model_params_count(model, trainable_only=False):
    """
    Print the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only (bool): If True, count only trainable params
    """
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        label = "Trainable"
    else:
        total = sum(p.numel() for p in model.parameters())
        label = "Total"

    print(f"{label} parameters: {total:,} ({total / 1e6:.2f}M)")


if __name__ == "__main__":
    from torch.utils.flop_counter import FlopCounterMode

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PixNerDiT().to(device=device, dtype=torch.float32)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    x = torch.randn(2, 3, 256, 256, dtype=torch.float32, device=device)
    t = torch.rand(2, dtype=torch.float32, device=device) * 1000
    y = torch.randint(0, 1000, (2, 16), dtype=torch.long, device=device)
    print_model_params_count(model.blocks)
    with FlopCounterMode(display=True):
        out = model(x, t, y)
