import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
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
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Pass the weight to the functional call
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)


Norm = RMSNorm


class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        nn.init.constant_(self.w2.weight, 0)
        self.act = ReLUSquared()

    def forward(self, x):
        return self.w2((F.relu(self.w1(x)) ** 2))


FeedForward = SwiGLU


class FeedForwardDW(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True)
        nn.init.constant_(self.w2.weight, 0)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x1 = self.w1(x)
        x1_spatial = x1.transpose(1, 2).reshape(B, -1, H, W)
        x1_spatial = self.dwconv(x1_spatial)
        x1 = x1_spatial.flatten(2).transpose(1, 2)

        return self.w2((F.relu(x1) ** 2))


class Embed(nn.Module):
    def __init__(self, in_features, out_features, bias=True, norm_layer=None):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)
        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            ReLUSquared(),
            nn.Linear(hidden_size, hidden_size, bias=True),
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
        self.qkv = nn.Linear(dim, dim * 4, bias=qkv_bias)
        self.lam_proj = nn.Linear(dim, num_heads, bias=True)

        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.lam_proj.weight, 0)
        nn.init.constant_(self.lam_proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.dim * 2, self.dim, self.dim], dim=-1)

        q = q.reshape(B, N, self.num_heads * 2, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())

        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)

        lam = self.lam_proj(x)
        lam = lam.transpose(1, 2).unsqueeze(-1)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn1 = attn[:, 0::2]
        attn2 = attn[:, 1::2]

        lam_val = torch.sigmoid(lam)
        x_out = attn1 - lam_val * attn2

        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=True)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_msa * attn_out

        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_out

        return x


class PixNerDiT(nn.Module):
    def __init__(
            self,
            in_channels=3,
            num_groups=16,
            hidden_size=1024,
            num_encoder_blocks=24,
            patch_size=16,
            vocab_size=12477,
            txt_max_length=32,
            weight_path=None,
            load_ema=False,
            route_ratio=0.75,
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
        self.route_ratio = route_ratio
        self.s_embedder = Embed(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.y_embedder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_encoder_blocks)
        ])

        self.final_conv = nn.Conv2d(self.hidden_size, self.in_channels * self.patch_size ** 2, kernel_size=3, padding=1)
        self.uncertainty_head = nn.Linear(self.hidden_size, 1)

        self.sink_tokens = nn.Parameter(torch.zeros(1, 4, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.fusion_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)

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
        nn.init.constant_(self.uncertainty_head.weight, 0)
        nn.init.constant_(self.uncertainty_head.bias, 0)
        nn.init.normal_(self.sink_tokens, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.constant_(self.fusion_proj.weight, 0)
        nn.init.constant_(self.fusion_proj.bias, 0)

    def forward(self, x, t, y, return_layers=None, path_drop=False):
        B, _, H, W = x.shape
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        x_unfolded = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        y_multi_hot = torch.zeros(B, self.vocab_size, dtype=x.dtype, device=y.device)
        y_multi_hot.scatter_(1, y, 1.0)
        y_emb = self.y_embedder(y_multi_hot).view(B, 1, self.hidden_size)

        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        t_emb = self.t_embedder(t)
        c = t_emb + y_emb

        seq = self.s_embedder(x_unfolded)

        num_blocks = len(self.blocks)
        layer_feats = {}

        # SPRINT Routing Setup
        route_start = 2
        route_end = num_blocks - 2
        route_ratio = 0.75
        routed_indices = None
        unrouted_indices = None
        seq_len = seq.shape[1]
        dense_features = None

        for i, block in enumerate(self.blocks):
            if i == route_start:
                dense_features = seq
                if path_drop:
                    pass
                elif self.training:
                    num_route = int(seq_len * route_ratio)
                    perm = torch.randperm(seq_len, device=seq.device)
                    routed_indices = perm[:num_route]
                    unrouted_indices = perm[num_route:]
                    seq = seq[:, unrouted_indices, :]

            if path_drop and route_start <= i < route_end:
                continue

            if i == route_end:
                if path_drop:
                    g_pad = self.mask_token.expand(B, seq_len, -1)
                    fused = torch.cat([dense_features, g_pad], dim=-1)
                    seq = self.fusion_proj(fused)
                else:
                    if self.training and routed_indices is not None:
                        g_pad = torch.zeros(B, seq_len, self.hidden_size, dtype=seq.dtype, device=seq.device)
                        r_idx_exp = routed_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.hidden_size)
                        u_idx_exp = unrouted_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.hidden_size)
                        mask_tokens = self.mask_token.expand(B, len(routed_indices), -1)
                        g_pad.scatter_(1, r_idx_exp, mask_tokens)
                        g_pad.scatter_(1, u_idx_exp, seq)
                        seq = g_pad
                    else:
                        seq = seq
                    fused = torch.cat([dense_features, seq], dim=-1)
                    seq = self.fusion_proj(fused)

            if self.grad_checkpointing and self.training:
                seq = checkpoint(block, seq, c, use_reentrant=False, context_fn=context_fn)
            else:
                seq = block(seq, c)

            if return_layers is not None and i in return_layers:
                layer_feats[i] = seq

        logvar_theta = self.uncertainty_head(seq)
        logvar_theta = logvar_theta.transpose(1, 2).reshape(B, 1, H_patch, W_patch)

        s = seq
        s = s.transpose(1, 2).reshape(B, self.hidden_size, H_patch, W_patch)
        x_out = self.final_conv(s)
        x_out = x_out.reshape(B, self.in_channels * self.patch_size ** 2, -1)
        x_out = torch.nn.functional.fold(x_out, (H, W), kernel_size=self.patch_size, stride=self.patch_size)

        if return_layers is not None:
            return x_out, logvar_theta, layer_feats
        return x_out, logvar_theta

    @torch.no_grad()
    def sample(self, B, H, W, device, steps=50, y=None, cfg_scale=4.0, pad_idx=None, sampler_type="euler", sampler_percentile=0.4, sampler_alpha=1.5, sampler_inner_steps=10):
        x = torch.randn(B, self.in_channels, H, W, device=device)
        dt = 1.0 / steps
        for step in range(steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device) * 1000

            if cfg_scale > 1.0 and y is not None:
                out_cond, uc_cond = self(x, t, y, path_drop=False)
                out_uncond, uc_uncond = self(x, t, y, path_drop=True)
                x0_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
                uc_pred = uc_cond
            else:
                x0_pred, uc_pred = self(x, t, y, path_drop=False)

            denom = max(1.0 - t_val, 0.05)
            v_pred = (x0_pred - x) / denom
            
            if sampler_type == "euler":
                x = x + v_pred * dt
                
            elif sampler_type == "look-ahead":
                uc_flat = uc_pred.reshape(B, -1)
                k = max(1, int(uc_flat.shape[1] * sampler_percentile))
                tau_p = torch.kthvalue(uc_flat, k, dim=1).values.view(B, 1, 1, 1)
                
                uc_upscaled = F.interpolate(uc_pred, scale_factor=self.patch_size, mode='nearest')
                M_conf = (uc_upscaled <= tau_p).float()
                M_unc = 1.0 - M_conf
                
                t_ctx_val = min(sampler_alpha * t_val, 1.0)
                if t_ctx_val <= t_val:
                    t_ctx_val = t_val + dt # always provide at least one step lookahead
                    
                x_ctx = x + v_pred * (t_ctx_val - t_val)
                x_mix = M_conf * x_ctx + M_unc * x
                
                t_mix_pixel = M_conf * t_ctx_val + M_unc * t_val
                t_mix_patch = F.interpolate(t_mix_pixel, scale_factor=1/self.patch_size, mode='nearest')
                t_mix_tensor = t_mix_patch * 1000
                
                if cfg_scale > 1.0 and y is not None:
                    out_cond, _ = self(x_mix, t_mix_tensor, y, path_drop=False)
                    out_uncond, _ = self(x_mix, t_mix_tensor, y, path_drop=True)
                    x0_mix = out_uncond + cfg_scale * (out_cond - out_uncond)
                else:
                    x0_mix, _ = self(x_mix, t_mix_tensor, y, path_drop=False)
                
                v_ctx = (x0_mix - x) / denom
                v_final = M_unc * v_ctx + M_conf * v_pred
                x = x + v_final * dt
                
            elif sampler_type == "dual-loop":
                uc_flat = uc_pred.reshape(B, -1)
                k = max(1, int(uc_flat.shape[1] * sampler_percentile))
                tau_p = torch.kthvalue(uc_flat, k, dim=1).values.view(B, 1, 1, 1)
                
                uc_upscaled = F.interpolate(uc_pred, scale_factor=self.patch_size, mode='nearest')
                M_conf = (uc_upscaled <= tau_p).float()
                M_unc = 1.0 - M_conf
                
                dt_inner = dt / sampler_inner_steps
                x_conf_next = x + v_pred * dt
                x_inner = x.clone()
                
                t_next_val = t_val + dt
                
                for inner in range(sampler_inner_steps):
                    t_inner_val = t_val + inner * dt_inner
                    t_mix_pixel = M_conf * t_next_val + M_unc * t_inner_val
                    t_mix_patch = F.interpolate(t_mix_pixel, scale_factor=1/self.patch_size, mode='nearest')
                    t_mix_tensor = t_mix_patch * 1000
                    
                    x_mix = M_conf * x_conf_next + M_unc * x_inner
                    
                    if cfg_scale > 1.0 and y is not None:
                        out_cond, _ = self(x_mix, t_mix_tensor, y, path_drop=False)
                        out_uncond, _ = self(x_mix, t_mix_tensor, y, path_drop=True)
                        x0_mix = out_uncond + cfg_scale * (out_cond - out_uncond)
                    else:
                        x0_mix, _ = self(x_mix, t_mix_tensor, y, path_drop=False)
                        
                    v_mix = (x0_mix - x_inner) / max(1.0 - t_inner_val, 0.05)
                    x_inner = x_inner + v_mix * dt_inner
                    
                x = M_conf * x_conf_next + M_unc * x_inner

        return x


from collections import defaultdict


def load_checkpoint(model, ckpt_path, device="cpu", ema_key=None):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Common checkpoint structures
    if isinstance(ckpt, dict):
        if ema_key is not None and ema_key in ckpt:
            state_dict = ckpt[ema_key]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Remove "module." prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    print(f"Loaded checkpoint: {ckpt_path}")

    if len(missing) > 0:
        print(f"\nMissing keys ({len(missing)}):")
        for k in missing:
            print("  ", k)

    if len(unexpected) > 0:
        print(f"\nUnexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print("  ", k)

    return model


def print_weight_statistics(model):
    print("\n================ WEIGHT STATS ================\n")

    total_nan = 0

    for name, param in model.named_parameters():
        if param.numel() == 0:
            continue

        data = param.data.float()

        nan_count = torch.isnan(data).sum().item()
        total_nan += nan_count

        print(
            f"{name:60s} "
            f"shape={str(tuple(data.shape)):20s} "
            f"mean={data.mean().item():+.5f} "
            f"std={data.std().item():.5f} "
            f"absmax={data.abs().max().item():.5f} "
            f"nan={nan_count}"
        )

    print(f"\nTotal NaNs in weights: {total_nan}")


def register_activation_hooks(model):
    activation_stats = {}

    def hook_fn(name):
        def fn(module, inp, out):
            with torch.no_grad():

                # Some modules return tuples
                if isinstance(out, tuple):
                    out = out[0]

                if not torch.is_tensor(out):
                    return

                out = out.float()

                activation_stats[name] = {
                    "shape": tuple(out.shape),
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "absmax": out.abs().max().item(),
                    "nan": torch.isnan(out).sum().item(),
                }

        return fn

    handles = []

    for name, module in model.named_modules():

        # Skip container modules
        if len(list(module.children())) > 0:
            continue

        handles.append(
            module.register_forward_hook(hook_fn(name))
        )

    return activation_stats, handles


def print_activation_statistics(stats):
    print("\n================ ACTIVATION STATS ================\n")

    total_nan = 0

    for name, s in stats.items():
        total_nan += s["nan"]

        print(
            f"{name:60s} "
            f"shape={str(s['shape']):20s} "
            f"mean={s['mean']:+.5f} "
            f"std={s['std']:.5f} "
            f"absmax={s['absmax']:.5f} "
            f"nan={s['nan']}"
        )

    print(f"\nTotal NaNs in activations: {total_nan}")


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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PixNerDiT().to(device=device, dtype=torch.float32)


    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params / 1e6:.2f} M")

    # Print weight stats
    print_weight_statistics(model)
