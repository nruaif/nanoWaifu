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
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.constant_(self.proj.weight, 0)

    def forward(self, x: torch.Tensor, pos, num_txt_tokens: int) -> torch.Tensor:
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

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=True)
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
        return x


class PixNerDiT(nn.Module):
    def __init__(
            self,
            in_channels=3,
            num_groups=16,
            hidden_size=1024,
            num_encoder_blocks=16,
            patch_size=16,
            vocab_size=12477,
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
        
        # DINO projection layers
        self.dino_proj = nn.Conv2d(self.hidden_size, 768, kernel_size=3, padding=1)
        self.dino_gamma = 0.8

        self.final_conv = nn.Conv2d(self.hidden_size, (self.in_channels) * self.patch_size ** 2, kernel_size=3, padding=1)

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
        
    def process_dino_features(self, dino_feat, H_patch, W_patch):
        # dino_feat: [B, 2057, 768] (with 5 cls/reg tokens)
        B = dino_feat.shape[0]
        # Remove first 5 tokens
        x = dino_feat[:, 5:, :]
        
        # Spatial normalization on encoder features [B, T, D]
        x = x - self.dino_gamma * x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        
        return x

    def forward(self, x, t, y, return_layer_4_feat=False):
        B, _, H, W = x.shape
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        x_unfolded = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        xpos = self.fetch_pos(H_patch, W_patch, x.device)
        ypos = self.y_pos_embedding[:, :y.shape[1], :]
        
        # Patch-level t's support
        if t.dim() == 1:
            # Broadcast scalar t to all patches if needed
            t = t.view(B, 1, 1).expand(B, H_patch * W_patch, 1)
        elif t.dim() == 2:
             t = t.unsqueeze(-1)
        elif t.dim() == 3:
            t = t.view(B, H_patch * W_patch, 1)

        t_emb = self.t_embedder(t) # (B, N, hidden_size)

        y_emb = self.y_embedder(y).view(B, -1, self.hidden_size)

        s = self.s_embedder(x_unfolded)
        s = s + t_emb

        num_txt_tokens = y_emb.shape[1]
        seq = torch.cat([y_emb, s], dim=1)

        num_blocks = len(self.blocks)
        layer_4_feat = None
        for i, block in enumerate(self.blocks):
            if self.grad_checkpointing and self.training:
                seq = checkpoint(block, seq, xpos, num_txt_tokens, H_patch, W_patch, use_reentrant=False,
                                        context_fn=context_fn)
            else:
                seq = block(seq, xpos, num_txt_tokens, H_patch, W_patch)
                
            if i == 3 and return_layer_4_feat:
                layer_4_feat = seq[:, num_txt_tokens:]

        s = seq[:, num_txt_tokens:]

        s = s.transpose(1, 2).reshape(B, self.hidden_size, H_patch, W_patch)
        x_out = self.final_conv(s) # (B, (in_channels + 1) * patch_size**2, H_patch, W_patch)
        
        # Split uncertainty
        c = self.in_channels * self.patch_size ** 2
        x_out = x_out[:, :c, :, :]
        
        x_out = x_out.reshape(B, self.in_channels * self.patch_size ** 2, -1)
        x_out = torch.nn.functional.fold(x_out, (H, W), kernel_size=self.patch_size, stride=self.patch_size)

        if return_layer_4_feat:
            layer_4_feat_spatial = layer_4_feat.transpose(1, 2).reshape(B, self.hidden_size, H_patch, W_patch)
            layer_4_feat_proj = self.dino_proj(layer_4_feat_spatial)
            layer_4_feat = layer_4_feat_proj.flatten(2).transpose(1, 2)
            return x_out, layer_4_feat
        return x_out

    @torch.no_grad()
    def sample(self, B, H, W, device, steps=50, y=None, cfg_scale=4.0, pad_idx=None, p=0.4, alpha=1.5):
        x = torch.randn(B, self.in_channels, H, W, device=device)
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        
        # Start at t=0 (full noise) and go to t=1 (clean) following flow matching
        # Wait, the original code had t=1000 to t=0 for diffusion.
        # But flow matching usually goes t=0 (noise) to t=1 (data), as defined in eq (1): x_t = t x_1 + (1-t) x_0.
        # The previous sample code had x = x + v_pred * dt, which is typical for Euler from t=0 to 1 if x0_pred is used to compute v_pred = (x0_pred - x) / denom.
        # Let's adapt to use the predicted velocity directly.
        # Original:
        # denom = max(1.0 - t_val, 0.05)
        # v_pred = (x0_pred - x) / denom
        # In Flow Matching, the network predicts the velocity v_theta(x_t, t) pointing from noise to data.
        
        dt = 1.0 / steps
        t_current = torch.zeros(B, H_patch * W_patch, 1, device=device) # Start at t=0

        for step in range(steps):
            t_next_val = (step + 1) * dt
            
            # 1. Predict velocity and uncertainty
            # t passed to forward should be t_current * 1000 if that's what TimestepEmbedder expects
            t_in = t_current * 1000
            
            if cfg_scale > 1.0 and y is not None and pad_idx is not None:
                x_in = x.repeat(2, 1, 1, 1)
                t_in = t_in.repeat(2, 1, 1)
                y_uncond = y.clone()
                y_uncond[:] = pad_idx
                y_in = torch.cat([y, y_uncond], dim=0)

                out, logvar = self(x_in, t_in, y_in)
                out_cond, out_uncond = out.chunk(2, dim=0)
                v_pred = out_uncond + cfg_scale * (out_cond - out_uncond) # Assuming output is x1 or velocity
                
                # If output is x1, we compute velocity:
                denom = torch.clamp(1.0 - t_current.view(B, 1, H_patch, W_patch), min=0.05)
                v_pred = (v_pred - x) / denom
                
                logvar_cond, logvar_uncond = logvar.chunk(2, dim=0)
                # Just use conditional uncertainty for thresholding
                uc = logvar_cond
            else:
                x1_pred, logvar = self(x, t_in, y)
                denom = torch.clamp(1.0 - t_current.view(B, 1, H_patch, W_patch), min=0.05)
                v_pred = (x1_pred - x) / denom
                uc = logvar
                
            # Average uncertainty over channels
            uc = uc.mean(dim=1, keepdim=True) # (B, 1, H_patch, W_patch)
            
            # 2. Adaptive thresholding
            # Flatten to find percentiles per item in batch
            uc_flat = uc.view(B, -1)
            # Find the value at percentile p (lower p = lower uncertainty = easier)
            # Higher uncertainty means harder. The paper says: "percentile for confident pixels... our samplers perform best at around the 40% percentile"
            # This means we take the bottom 40% of uncertainty values as confident.
            k = int(p * uc_flat.shape[1])
            k = max(1, min(k, uc_flat.shape[1] - 1))
            tau_p = torch.kthvalue(uc_flat, k, dim=1).values.view(B, 1, 1, 1)
            
            M_conf = (uc <= tau_p).float()
            M_unc = 1.0 - M_conf
            
            # 3. Look-ahead for context
            # advance confident patches by alpha * dt, capped at 1.0
            t_ctx = torch.clamp(t_current.view(B, 1, H_patch, W_patch) + alpha * dt, max=1.0)
            x_ctx = x + (t_ctx - t_current.view(B, 1, H_patch, W_patch)) * v_pred
            
            # Create mixed input x_tilde
            x_tilde = M_conf * x_ctx + M_unc * x
            t_tilde = M_conf * t_ctx + M_unc * t_current.view(B, 1, H_patch, W_patch)
            t_tilde_flat = t_tilde.view(B, H_patch * W_patch, 1)
            
            # 4. Context-aware velocity
            t_in_tilde = t_tilde_flat * 1000
            
            if cfg_scale > 1.0 and y is not None and pad_idx is not None:
                x_in_tilde = x_tilde.repeat(2, 1, 1, 1)
                t_in_tilde = t_in_tilde.repeat(2, 1, 1)
                
                out_tilde, _ = self(x_in_tilde, t_in_tilde, y_in)
                out_cond_tilde, out_uncond_tilde = out_tilde.chunk(2, dim=0)
                v_ctx_pred = out_uncond_tilde + cfg_scale * (out_cond_tilde - out_uncond_tilde)
                
                denom_tilde = torch.clamp(1.0 - t_tilde, min=0.05)
                v_ctx_pred = (v_ctx_pred - x_tilde) / denom_tilde
            else:
                x1_tilde, _ = self(x_tilde, t_in_tilde, y)
                denom_tilde = torch.clamp(1.0 - t_tilde, min=0.05)
                v_ctx_pred = (x1_tilde - x_tilde) / denom_tilde
                
            # Replace uncertain prediction
            v_final = M_unc * v_ctx_pred + M_conf * v_pred
            
            # 5. Advance all to t_next
            x = x + (t_next_val - t_current.view(B, 1, H_patch, W_patch)) * v_final
            
            t_current = torch.full((B, H_patch * W_patch, 1), t_next_val, device=device)

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

    ckpt_path = "ckpt_step_185000.pth"

    load_checkpoint(
        model,
        ckpt_path,
        device=device,
        ema_key=None,  # e.g. "ema" if your checkpoint stores EMA weights
    )

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params / 1e6:.2f} M")

    # Print weight stats
    print_weight_statistics(model)

    # Register activation hooks
    activation_stats, handles = register_activation_hooks(model)

    # Single-image forward pass
    x = torch.randn(1, 3, 256, 256, dtype=torch.float32, device=device)

    t = torch.rand(1, dtype=torch.float32, device=device) * 1000

    y = torch.randint(
        0,
        1000,
        (1, 16),
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        out = model(x, t, y)

    print("\nOutput stats:")
    print(
        f"shape={tuple(out.shape)} "
        f"mean={out.mean().item():+.5f} "
        f"std={out.std().item():.5f} "
        f"absmax={out.abs().max().item():.5f}"
    )

    # Print activation stats
    print_activation_statistics(activation_stats)

    # Cleanup hooks
    for h in handles:
        h.remove()