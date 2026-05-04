import torch
import torch.nn as nn
import torch.functional as F
import math
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, CheckpointPolicy, create_selective_checkpoint_contexts
from tqdm.auto import tqdm
from torch.utils.flop_counter import FlopCounterMode
import functools

aten = torch.ops.aten
compute_intensive_ops = [
    aten.mm.default,
    aten.bmm,
    aten.addmm,
]


def policy_fn(ctx, op, *args, **kwargs):
    if op in compute_intensive_ops:
        return CheckpointPolicy.MUST_SAVE
    else:
        return CheckpointPolicy.PREFER_RECOMPUTE


context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input.contiguous())
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def uniformity_loss(x, sketch_dim=64):
    """
    Forces Covariance(x) ~ Identity.
    Matches the 2nd Moment (Spherical Cloud).
    """
    if dist.is_initialized():
        x = torch.cat(GatherLayer.apply(x), dim=0)

    N, C = x.size()
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device, dtype=x.dtype) / (C ** 0.5)
        x = x @ S.T
    else:
        sketch_dim = C

    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)

    target = torch.eye(sketch_dim, device=x.device)

    return torch.norm(cov - target, p='fro')


class MeanPoolingEmbedder(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.embed = nn.EmbeddingBag(num_classes + 1, dim * 4, mode='mean')

    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor) -> torch.Tensor:
        x = self.embed(y_indices, y_offsets)
        return x.view(x.shape[0], 4, -1)


class GGRoPE2d(nn.Module):
    def __init__(
            self,
            n_heads: int,
            head_dim: int,
            min_freq: float,
            max_freq: float,
            p_zero_freqs: float = 0.0,
            direction_spacing: float = math.pi * (math.sqrt(5) - 1) / 2,
    ):
        super().__init__()
        assert head_dim % 2 == 0
        assert 0 <= p_zero_freqs <= 1
        self.n_heads = n_heads
        self.head_dim = head_dim
        n_freqs = head_dim // 2
        n_zero_freqs = round(p_zero_freqs * n_freqs)

        omega_F = torch.cat(
            (
                torch.zeros(n_zero_freqs),
                min_freq
                * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
            )
        )
        phi_hF = (
                torch.arange(n_heads * n_freqs).reshape(n_heads, n_freqs)
                * direction_spacing
        )
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
        freqs_hF2 = omega_F.unsqueeze(-1) * directions_hF2
        self.register_buffer("freqs_hF2", freqs_hF2)

    def forward(self, x: torch.Tensor, H: int, W: int, seq_indices: torch.Tensor = None) -> torch.Tensor:
        B, h, N, d = x.shape

        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_grid = torch.linspace(-xlim, xlim, W, device=x.device, dtype=x.dtype)
        y_grid = torch.linspace(-ylim, ylim, H, device=x.device, dtype=x.dtype)

        y_HW, x_HW = torch.meshgrid(y_grid, x_grid, indexing='ij')
        positions_HW2 = torch.stack((x_HW, y_HW), dim=-1).reshape(H * W, 1, 1, 2)

        theta = (self.freqs_hF2 * positions_HW2).sum(dim=-1)

        cos = torch.cos(theta).permute(1, 0, 2).unsqueeze(0)
        sin = torch.sin(theta).permute(1, 0, 2).unsqueeze(0)

        if seq_indices is not None:
            cos = torch.gather(cos.expand(B, -1, -1, -1), 2,
                               seq_indices.unsqueeze(1).unsqueeze(-1).expand(-1, h, -1, cos.shape[-1]))
            sin = torch.gather(sin.expand(B, -1, -1, -1), 2,
                               seq_indices.unsqueeze(1).unsqueeze(-1).expand(-1, h, -1, sin.shape[-1]))

        x_fp32 = x.float()
        x1, x2 = x_fp32.chunk(2, dim=-1)

        x_out1 = x1 * cos - x2 * sin
        x_out2 = x1 * sin + x2 * cos

        output = torch.cat((x_out1, x_out2), dim=-1)
        return output.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), weight=self.weight, eps=self.eps)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)
        return self.fc2(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, use_value_residual=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

        self.use_value_residual = use_value_residual
        if self.use_value_residual:
            self.lambda_1 = nn.Parameter(torch.tensor(0.5))
            self.lambda_2 = nn.Parameter(torch.tensor(0.5))

        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, H, W, rope, num_context_tokens=0, seq_indices=None, v1=None):
        B, N_seq, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N_seq, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if num_context_tokens > 0:
            c_q, s_q = q[:, :, :num_context_tokens, :], q[:, :, num_context_tokens:, :]
            c_k, s_k = k[:, :, :num_context_tokens, :], k[:, :, num_context_tokens:, :]
            s_q = rope(s_q, H, W, seq_indices)
            s_k = rope(s_k, H, W, seq_indices)
            q = torch.cat([c_q, s_q], dim=2)
            k = torch.cat([c_k, s_k], dim=2)
        else:
            q = rope(q, H, W, seq_indices)
            k = rope(k, H, W, seq_indices)

        # Value Residual Learning (ResFormer pattern)
        if self.use_value_residual and v1 is not None:
            v = self.lambda_1 * v1 + self.lambda_2 * v

        dtype = q.dtype
        x_att = F.scaled_dot_product_attention(q, k, v)
        x_att = x_att.to(dtype).transpose(1, 2).reshape(B, N_seq, C)

        out = self.proj(x_att)

        # If this is the layer designated to produce v1, return it
        if not self.use_value_residual and v1 is None:
            return out, v

        return out, None


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, use_value_residual=False):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, use_value_residual=use_value_residual)

        self.norm3 = RMSNorm(dim)
        self.mlp = Mlp(dim, dim * 4, dim)

    def forward(self, x, H, W, rope, context=None, seq_indices=None, v1=None):
        B, N, C = x.shape

        num_context_tokens = 0
        if context is not None:
            x_seq = torch.cat([context, x], dim=1)
            num_context_tokens = context.shape[1]
        else:
            x_seq = x

        # ---- Self-attention ----
        x_norm1 = self.norm1(x_seq)
        attn_out, new_v1 = self.self_attn(
            x_norm1, H, W, rope, num_context_tokens, seq_indices, v1=v1
        )
        x_seq = x_seq + attn_out

        # ---- MLP ----
        x_seq = x_seq + self.mlp(self.norm3(x_seq))

        # ---- Split back context ----
        if context is not None:
            context = x_seq[:, :num_context_tokens, :]
            x = x_seq[:, num_context_tokens:, :]
        else:
            x = x_seq

        return x, context, new_v1


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
        t = t * 1000.0
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        if t.dim() == 1:
            emb = t.float()[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).to(self.mlp[0].weight.dtype)
            return self.mlp(emb)
        elif t.dim() == 2:
            emb = t.float().unsqueeze(-1) * emb.view(1, 1, -1)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1).to(self.mlp[0].weight.dtype)
            return self.mlp(emb)


class TokenformerDiT(nn.Module):
    """
    Pure DiT architecture with SPRINT (Sparse-Dense Residual Fusion).
    Conditions via concatenated tokens in self-attention.
    Includes ResFormer Value Residual Learning for the second half of middle blocks.
    """

    def __init__(self, in_channels=32, dim=768, depth=12, num_heads=12,
                 attn_kv_pairs=576, ffn_kv_pairs=2304, num_classes=12476,
                 use_checkpoint=False, drop_ratio=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.drop_ratio = drop_ratio
        self.in_channels = in_channels

        self.patch_embed = nn.Linear(in_channels, dim)

        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = MeanPoolingEmbedder(num_classes, dim)

        self.rope = GGRoPE2d(
            n_heads=num_heads,
            head_dim=dim // num_heads,
            min_freq=1.0,
            max_freq=100.0
        )

        self.num_encoder_blocks = 2
        self.num_decoder_blocks = 2
        self.num_middle_blocks = depth - self.num_encoder_blocks - self.num_decoder_blocks

        # U-Net style value residual application
        self.mid_block_halfway = self.num_middle_blocks // 2

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, use_value_residual=False) for _ in range(self.num_encoder_blocks)
        ])

        # Apply value residual only to the second half of the middle blocks
        self.middle_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, use_value_residual=(i >= self.mid_block_halfway))
            for i in range(self.num_middle_blocks)
        ])

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, use_value_residual=False) for _ in range(self.num_decoder_blocks)
        ])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.fusion = nn.Linear(dim * 2, dim)

        self.final_norm = RMSNorm(dim)
        self.final_proj = nn.Linear(dim, in_channels + 1)

        self.w_k = nn.Parameter(torch.tensor(0.0))
        self.refiner = nn.Conv2d(in_channels + 1, in_channels + 1, 3, 1, 1)

    def get_velocity(self, z, u_pred, t):
        k = torch.sigmoid(self.w_k)
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        elif t.dim() == 2:
            B, N = t.shape
            _, _, H, W = z.shape
            t = t.view(B, 1, H, W)
        denom = (k * t + (1 - k) * (1 - t)).clamp(min=0.05)
        return -((1 - 2 * k) * z + u_pred) / denom

    def forward(self, x_in, t, y_indices, y_offsets=None, return_features=False, path_drop_prob=0.0,
                force_path_drop=False, return_uniformity=False, force_drop=False):
        B, C, H, W = x_in.shape
        N = H * W
        x = x_in.flatten(2).transpose(1, 2)
        x = self.patch_embed(x)

        if t.dim() == 1:
            t = t.unsqueeze(1).expand(-1, N)
        t_embeds = self.t_embedder(t)
        x = x + t_embeds

        y_tokens = self.y_embedder(y_indices, y_offsets)
        context = y_tokens

        # Variables for Value Residual Tracking
        v_first_mid = None
        v_current = None

        for block in self.encoder_blocks:
            if self.use_checkpoint and self.training:
                # Modifying checkpointing signature to handle v1 parameter passing
                def block_wrapper(b, x_in, ctx_in):
                    x_out, ctx_out, _ = b(x_in, H, W, self.rope, ctx_in, None, v1=None)
                    return x_out, ctx_out

                x, context = checkpoint(block_wrapper, block, x, context, use_reentrant=False, context_fn=context_fn)
            else:
                x, context, _ = block(x, H, W, self.rope, context, None, v1=None)

        f_t = x

        do_drop = False
        if self.drop_ratio > 0.0:
            if force_drop:
                do_drop = True
            elif self.training:
                if torch.rand(1).item() > 0.2:
                    do_drop = True

        if do_drop:
            num_keep = int(N * (1 - self.drop_ratio))
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            seq_indices = ids_shuffle[:, :num_keep]
            x_middle = torch.gather(x, dim=1, index=seq_indices.unsqueeze(-1).expand(-1, -1, self.dim))
        else:
            x_middle = x
            seq_indices = None

        if force_path_drop:
            g_pad_t = self.mask_token.expand(B, N, -1)
            u_loss = torch.tensor(0.0, device=x.device) if return_uniformity else None
        else:
            for idx, block in enumerate(self.middle_blocks):

                # First middle block creates the base V_1 for ResFormer
                if idx == 0:
                    if self.use_checkpoint and self.training:
                        x_middle, context, v_current = block(x_middle, H, W, self.rope, context, seq_indices, v1=None)
                    else:
                        x_middle, context, v_current = block(x_middle, H, W, self.rope, context, seq_indices, v1=None)
                    v_first_mid = v_current

                # Second half of middle blocks applies ResFormer
                elif idx >= self.mid_block_halfway:
                    if self.use_checkpoint and self.training:
                        def block_wrapper_v1(b, x_in, ctx_in, seq_in, v_in):
                            x_out, ctx_out, _ = b(x_in, H, W, self.rope, ctx_in, seq_in, v1=v_in)
                            return x_out, ctx_out

                        x_middle, context = checkpoint(block_wrapper_v1, block, x_middle, context, seq_indices,
                                                       v_first_mid, use_reentrant=False, context_fn=context_fn)
                    else:
                        x_middle, context, _ = block(x_middle, H, W, self.rope, context, seq_indices, v1=v_first_mid)

                # Normal processing for first half of middle blocks
                else:
                    if self.use_checkpoint and self.training:
                        def block_wrapper_norm(b, x_in, ctx_in, seq_in):
                            x_out, ctx_out, _ = b(x_in, H, W, self.rope, ctx_in, seq_in, v1=None)
                            return x_out, ctx_out

                        x_middle, context = checkpoint(block_wrapper_norm, block, x_middle, context, seq_indices,
                                                       use_reentrant=False, context_fn=context_fn)
                    else:
                        x_middle, context, _ = block(x_middle, H, W, self.rope, context, seq_indices, v1=None)

            u_loss = None
            if return_uniformity and self.training:
                x_mid_pooled = x_middle.mean(dim=1)
                u_loss = uniformity_loss(x_mid_pooled)
            elif return_uniformity:
                u_loss = torch.tensor(0.0, device=x.device)

            g_full = self.mask_token.expand(B, N, -1).clone()
            if seq_indices is not None:
                g_full.scatter_(1, seq_indices.unsqueeze(-1).expand(-1, -1, self.dim), x_middle)
            else:
                g_full = x_middle

            if self.training and path_drop_prob > 0.0:
                drop_mask = (torch.rand(1, device=x.device) < path_drop_prob).view(1, 1, 1)
                g_pad_t = torch.where(drop_mask, self.mask_token.expand(B, N, -1), g_full)
            else:
                g_pad_t = g_full

        h_t = self.fusion(torch.cat([f_t, g_pad_t], dim=-1))

        for block in self.decoder_blocks:
            if self.use_checkpoint and self.training:
                def block_wrapper_dec(b, x_in, ctx_in):
                    x_out, ctx_out, _ = b(x_in, H, W, self.rope, ctx_in, None, v1=None)
                    return x_out, ctx_out

                h_t, context = checkpoint(block_wrapper_dec, block, h_t, context, use_reentrant=False,
                                          context_fn=context_fn)
            else:
                h_t, context, _ = block(h_t, H, W, self.rope, context, None, v1=None)

        feat = h_t
        x_norm = self.final_norm(h_t)

        x_out = self.final_proj(x_norm)

        x_out = x_out.transpose(1, 2).reshape(B, self.in_channels + 1, H, W)
        x_out = self.refiner(x_out)
        u_pred = x_out[:, :self.in_channels, :, :]
        logvar_theta = x_out[:, self.in_channels:, :, :]

        res = [u_pred, logvar_theta]
        if return_features:
            res.append(feat)
        if return_uniformity:
            res.append(u_loss)

        if len(res) == 2:
            return res[0], res[1]
        return tuple(res)


import random


class TagProcessor:
    def __init__(self, tags_file):
        with open(tags_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.num_classes = len(self.tags)

    def process_prompts(self, prompts, device, dropout_prob=0.0):
        indices = []
        offsets = [0]
        for p in prompts:
            if random.random() < dropout_prob:
                indices.append(self.num_classes)
            else:
                tags = p.split()
                count = 0
                for t in tags:
                    if t in self.tag_to_idx:
                        indices.append(self.tag_to_idx[t])
                        count += 1
                if count == 0:
                    indices.append(self.num_classes)
            offsets.append(len(indices))

        indices = torch.tensor(indices, dtype=torch.long, device=device)
        offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return indices, offsets


@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=50, guidance_scale=1.5, noise=None, cfg_scale=0,
                sampler_type="look-ahead", p_percentile=0.4, alpha=2.0):
    in_channels = model.in_channels if hasattr(model, 'in_channels') else (
        model.module.in_channels if hasattr(model, 'module') else 32)
    model.eval()
    if isinstance(latent_size, (tuple, list)):
        H, W = latent_size
    else:
        H = W = latent_size

    if noise is not None:
        if isinstance(noise, list):
            x = torch.stack([n.to(device) for n in noise[:batch_size]])
        else:
            x = noise.clone().to(device)
    else:
        x = torch.randn(batch_size, in_channels, H, W, device=device)

    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)
    null_prompts = [""] * batch_size
    y_null_indices, y_null_offsets = tag_processor.process_prompts(null_prompts, device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    x = x.to(torch.bfloat16)
    for i in tqdm(range(steps)):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size, H * W), t_curr.item(), device=device, dtype=x.dtype)

        if sampler_type == "euler":
            u_cond, logvar_cond = model(x, t_vec, y_indices, y_offsets)
            v_cond = model.get_velocity(x, u_cond, t_vec)

            u_uncond, _ = model(x, t_vec, y_null_indices, y_null_offsets, force_drop=True)
            v_uncond = model.get_velocity(x, u_uncond, t_vec)

            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            x = x + dt * v

        elif sampler_type == "look-ahead":
            u_cond, logvar_cond = model(x, t_vec, y_indices, y_offsets)
            v_cond = model.get_velocity(x, u_cond, t_vec)

            u_uncond, _ = model(x, t_vec, y_null_indices, y_null_offsets, force_drop=True)
            v_uncond = model.get_velocity(x, u_uncond, t_vec)

            v = v_uncond + guidance_scale * (v_cond - v_uncond)

            uc = logvar_cond
            uc_flat = uc.view(batch_size, -1)
            tau_p = torch.quantile(uc_flat.float(), p_percentile, dim=1, keepdim=True).to(x.dtype)
            tau_p = tau_p.view(batch_size, 1, 1, 1)

            M_conf = (uc <= tau_p).to(x.dtype)
            M_unc = 1.0 - M_conf

            t_ctx_val = max(t_curr.item() + alpha * dt.item(), 0.0)

            x_ctx = x + (t_ctx_val - t_curr.item()) * v
            x_tilde = M_conf * x_ctx + M_unc * x

            t_tilde_vec = M_conf.view(batch_size, -1) * t_ctx_val + M_unc.view(batch_size, -1) * t_curr.item()

            u_ctx_cond, _ = model(x_tilde, t_tilde_vec, y_indices, y_offsets)
            v_ctx_cond = model.get_velocity(x_tilde, u_ctx_cond, t_tilde_vec)

            u_ctx_uncond, _ = model(x_tilde, t_tilde_vec, y_null_indices, y_null_offsets, force_drop=True)
            v_ctx_uncond = model.get_velocity(x_tilde, u_ctx_uncond, t_tilde_vec)

            v_ctx = v_ctx_uncond + guidance_scale * (v_ctx_cond - v_ctx_uncond)

            v_final = M_unc * v_ctx + M_conf * v
            x = x + dt * v_final

    model.train()
    return x


if __name__ == "__main__":
    print("Initializing DiT on CPU...")
    device = torch.device("cpu")

    model = TokenformerDiT(
        in_channels=32,
        dim=768,
        depth=24,  # 24 total layers. 2 enc, 2 dec, 20 mid blocks. Value res applied to mid blocks 10-19.
        num_heads=12,
        num_classes=12476,
        drop_ratio=0.75
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params / 1e6:.2f} M")

    batch_size = 2
    latent_size = 16

    x_in = torch.randn(batch_size, 32, latent_size, latent_size, device=device)
    t = torch.rand(batch_size, device=device)

    y_indices = torch.randint(0, 12476, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)

    print("\nRunning forward pass (training)...")
    model.eval()
    with FlopCounterMode(model) as flop_counter:
        u_pred, logvar, u_loss = model(x_in, t, y_indices, y_offsets, path_drop_prob=0.1, return_uniformity=True)
        total_flops = flop_counter.get_total_flops()

    print(f"Input shape:  {x_in.shape}")
    print(f"Output u_pred shape: {u_pred.shape}")
    print(f"Output logvar shape: {logvar.shape}")
    print(total_flops)
    print("✅ Forward pass completed successfully!")