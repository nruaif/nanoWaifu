import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
import random

torch._dynamo.config.recompile_limit = 128


def get_timestep_embedding(t, dim):
    """Sinusoidal timestep embedding — replaces TimestepEmbedder class."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# =============================================================================
# FCDM Components (ConvNeXt-based conditioning)
# =============================================================================
class GlobalResponseNorm(nn.Module):
    """Global Response Normalization (GRN) from ConvNeXt V2."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x: [B, H, W, C]
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
        return x


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for FCDM."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim, bias=True)
        )
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, x, c):
        # x: [B, H, W, C], c: [B, cond_dim]
        x_norm = self.norm(x)
        modulation = self.mlp(c)
        gamma, beta, alpha = modulation.chunk(3, dim=-1)
        gamma = gamma.unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1)
        alpha = alpha.unsqueeze(1).unsqueeze(1)
        x_out = x_norm * (1 + gamma) + beta
        return x_out, alpha


class FCDMBlock(nn.Module):
    """FCDM Block: ConvNeXt with AdaLN conditioning."""
    def __init__(self, dim, cond_dim, expansion_ratio=3, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.adaln = AdaLN(dim, cond_dim)
        self.pwconv_expand = nn.Linear(dim, dim * expansion_ratio)
        self.grn = GlobalResponseNorm(dim * expansion_ratio)
        self.act = nn.GELU()
        self.pwconv_reduce = nn.Linear(dim * expansion_ratio, dim)
        nn.init.zeros_(self.pwconv_reduce.weight)
        nn.init.zeros_(self.pwconv_reduce.bias)

    def forward(self, x, c):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x, alpha = self.adaln(x, c)
        x = self.pwconv_expand(x)
        x = self.grn(x)
        x = self.act(x)
        x = self.pwconv_reduce(x)
        identity = identity.permute(0, 2, 3, 1)
        x = identity + alpha * x
        x = x.permute(0, 3, 1, 2)
        return x


class FCDMConditioning(nn.Module):
    """FCDM-based conditioning — drop-in replacement for TagTransformer."""
    def __init__(self, num_classes: int, dim: int, max_tags: int = 64,
                 num_fcdm_blocks: int = 2, expansion_ratio: int = 3):
        super().__init__()
        self.dim = dim
        self.max_tags = max_tags
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.padding_idx = num_classes
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.use_fcdm_blocks = num_fcdm_blocks > 0
        if self.use_fcdm_blocks:
            grid_size = int(math.sqrt(max_tags))
            self.tag_grid_size = grid_size
            self.fcdm_blocks = nn.ModuleList([
                FCDMBlock(dim, dim, expansion_ratio=expansion_ratio, kernel_size=3)
                for _ in range(num_fcdm_blocks)
            ])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.output_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        else:
            self.output_proj = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
            )

    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor = None) -> torch.Tensor:
        device = y_indices.device
        if y_offsets is None:
            B = 1
            tags = y_indices[:self.max_tags]
            if len(tags) < self.max_tags:
                tags = F.pad(tags, (0, self.max_tags - len(tags)), value=self.padding_idx)
            padded_tags = tags.unsqueeze(0)
        else:
            B = len(y_offsets)
            offsets = y_offsets.tolist()
            total_len = len(y_indices)
            lengths = [offsets[i + 1] - offsets[i] if i + 1 < B else total_len - offsets[i] for i in range(B)]
            padded_tags = torch.full((B, self.max_tags), self.padding_idx, dtype=torch.long, device=device)
            for i, length in enumerate(lengths):
                start = offsets[i]
                copy_len = min(length, self.max_tags)
                if copy_len > 0:
                    padded_tags[i, :copy_len] = y_indices[start:start + copy_len]

        tag_embeds = self.embedding(padded_tags)
        mask = (padded_tags != self.padding_idx).float().unsqueeze(-1)

        if self.use_fcdm_blocks:
            grid_size = self.tag_grid_size
            pad_len = grid_size * grid_size - self.max_tags
            if pad_len > 0:
                pad = torch.zeros(B, pad_len, self.dim, device=device)
                tag_embeds = torch.cat([tag_embeds, pad], dim=1)
                mask_pad = torch.zeros(B, pad_len, 1, device=device)
                mask = torch.cat([mask, mask_pad], dim=1)
            tag_embeds = tag_embeds.transpose(1, 2).reshape(B, self.dim, grid_size, grid_size)
            valid_embeds = (tag_embeds.reshape(B, self.dim, -1) * mask.reshape(B, 1, -1)).sum(dim=-1) / (mask.sum(dim=1) + 1e-6)
            c = valid_embeds
            x = tag_embeds
            for block in self.fcdm_blocks:
                x = block(x, c)
            x = self.pool(x).flatten(1)
            output = self.output_proj(x)
        else:
            masked_embeds = tag_embeds * mask
            sum_embeds = masked_embeds.sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            pooled = sum_embeds / count
            output = self.output_proj(pooled)
        return output



# =============================================================================
# Core Model Components
# =============================================================================
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

        omega_F = torch.cat((
            torch.zeros(n_zero_freqs),
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
        ))
        phi_hF = (
                torch.arange(n_heads * n_freqs).reshape(n_heads, n_freqs)
                * direction_spacing
        )
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
        freqs_hF2 = omega_F.unsqueeze(-1) * directions_hF2
        self.register_buffer("freqs_hF2", freqs_hF2)

    def forward(self, x: torch.Tensor, H: int, W: int, seq_indices: torch.Tensor = None) -> torch.Tensor:
        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_grid = torch.linspace(-xlim, xlim, W, device=x.device, dtype=x.dtype)
        y_grid = torch.linspace(-ylim, ylim, H, device=x.device, dtype=x.dtype)

        y_HW, x_HW = torch.meshgrid(y_grid, x_grid, indexing='ij')
        positions_HW2 = torch.stack((x_HW, y_HW), dim=-1).reshape(H * W, 1, 1, 2)

        theta = (self.freqs_hF2 * positions_HW2).sum(dim=-1)

        if seq_indices is not None:
            theta = theta[seq_indices].permute(0, 2, 1, 3)
        else:
            theta = theta.permute(1, 0, 2).unsqueeze(0)

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        x_fp32 = x.float()
        x1, x2 = x_fp32.chunk(2, dim=-1)

        x_out1 = x1 * cos - x2 * sin
        x_out2 = x1 * sin + x2 * cos

        return torch.cat((x_out1, x_out2), dim=-1).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.elementwise_affine = elementwise_affine
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), weight=self.weight, eps=self.eps)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, use_dwconv=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.use_dwconv = use_dwconv

        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.act = nn.GELU(approximate="tanh")
        if self.use_dwconv:
            self.dwconv = nn.Conv2d(
                hidden_features, hidden_features,
                kernel_size=3, padding=1, groups=hidden_features
            )
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)
        if self.use_dwconv:
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.dwconv(x)
            x = x.reshape(B, C, N).transpose(1, 2)
        return self.fc2(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

    def forward(self, x, H=None, W=None, rope=None, num_context_tokens=0, seq_indices=None, src_key_padding_mask=None):
        B, N_seq, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N_seq, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
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

        if src_key_padding_mask is not None:
            attn_mask = ~src_key_padding_mask.view(B, 1, 1, N_seq)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, N_seq, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, use_dwconv=True):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.self_attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        self.mlp = Mlp(dim, dim * 4, dim, use_dwconv=use_dwconv)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, H=None, W=None, rope=None, num_cls_tokens=0, src_key_padding_mask=None):
        if c.dim() == 2:
            c = c.unsqueeze(1)

        modulation = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1 * (1 + scale_msa) + shift_msa
        attn_out = self.self_attn(x_norm1, H, W, rope, num_context_tokens=num_cls_tokens,
                                  src_key_padding_mask=src_key_padding_mask)
        x = x + gate_msa * attn_out

        if num_cls_tokens > 0:
            cls = x[:, :num_cls_tokens]
            spatial = x[:, num_cls_tokens:]
            if shift_mlp.shape[1] > 1:
                shift_mlp = shift_mlp[:, num_cls_tokens:]
                scale_mlp = scale_mlp[:, num_cls_tokens:]
                gate_mlp = gate_mlp[:, num_cls_tokens:]
        else:
            spatial = x

        spatial_norm = self.norm2(spatial)
        spatial_norm = spatial_norm * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(spatial_norm, H, W)
        spatial = spatial + gate_mlp * mlp_out

        if num_cls_tokens > 0:
            x = torch.cat([cls, spatial], dim=1)
        else:
            x = spatial

        return x


class TokenformerDiT(nn.Module):
    """
    Diffusion Transformer (DiT) with RoPE, DW-Conv MLP, and learnable CLS tokens.
    Now supports FCDMConditioning as drop-in replacement for TagTransformer.
    Also supports return_layer_match for InfoNCE-style contrastive learning.
    """

    def __init__(self, in_channels=128, dim=768, depth=12, num_heads=12,
                 num_classes=12477, use_checkpoint=False, num_cls_tokens=4, max_tags=64,
                 fcdm_blocks=2):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.num_cls_tokens = num_cls_tokens

        self.patch_embed = nn.Linear(in_channels, dim)

        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, dim))
        nn.init.normal_(self.cls_tokens, std=0.02)

        # Conditioning module
        self.y_embedder = FCDMConditioning(
            num_classes=num_classes, dim=dim, max_tags=max_tags,
            num_fcdm_blocks=fcdm_blocks
        )

        self.rope = GGRoPE2d(
            n_heads=num_heads,
            head_dim=dim // num_heads,
            min_freq=1.0,
            max_freq=100.0
        )

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads) for _ in range(depth)
        ])

        self.final_norm = RMSNorm(dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        self.final_proj = nn.Linear(dim, in_channels)

        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)

    def forward(self, x_in, t, y_indices, y_offsets=None, return_features=False,
                return_layer_match=False, **kwargs):
        B, C, H, W = x_in.shape
        N = H * W

        x = x_in.flatten(2).transpose(1, 2)
        x = self.patch_embed(x)

        cls = self.cls_tokens.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        t_emb = get_timestep_embedding(t * 1000.0, self.dim).to(x_in.dtype).unsqueeze(1)
        y_emb = self.y_embedder(y_indices, y_offsets).unsqueeze(1)

        c_full = t_emb + y_emb
        c_time_only = t_emb

        # Store intermediate features for layer match loss
        layer_features = [] if return_layer_match else None

        for idx, block in enumerate(self.blocks):
            c = c_time_only if idx < 4 else c_full

            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, c, H, W, self.rope, self.num_cls_tokens,
                               use_reentrant=False)
            else:
                x = block(x, c, H, W, self.rope, num_cls_tokens=self.num_cls_tokens)

            if return_layer_match and idx % 4 == 3:  # every 4th block
                layer_features.append(x[:, self.num_cls_tokens:].mean(dim=1))  # [B, dim]

        x = x[:, self.num_cls_tokens:]

        final_mod = self.final_modulation(c_full)
        shift, scale = final_mod.chunk(2, dim=-1)

        x_norm = self.final_norm(x)
        x_norm = x_norm * (1 + scale) + shift
        x_out = self.final_proj(x_norm)

        x0_pred = x_out.transpose(1, 2).reshape(B, self.in_channels, H, W)

        # Compute InfoNCE-style layer match loss
        infonce_loss = torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)
        if return_layer_match and len(layer_features) >= 2:
            # Contrast: adjacent layer features should be similar, non-adjacent dissimilar
            for i in range(len(layer_features) - 1):
                pos_sim = F.cosine_similarity(layer_features[i], layer_features[i+1], dim=-1).mean()
                # Negative: random pair
                neg_idx = torch.randperm(B)
                neg_sim = F.cosine_similarity(layer_features[i], layer_features[i+1][neg_idx], dim=-1).mean()
                infonce_loss = infonce_loss + F.relu(neg_sim - pos_sim + 0.5)
            infonce_loss = infonce_loss / (len(layer_features) - 1)

        if return_features:
            return x0_pred, x
        if return_layer_match:
            return x0_pred, infonce_loss
        return x0_pred


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
                steps=250, guidance_scale=1.5, noise=None):
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
    for i in tqdm(range(steps), desc="Sampling", disable=device.type == 'cpu'):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size,), t_curr.item(), device=device, dtype=x.dtype)

        x0_cond = model(x, t_vec, y_indices, y_offsets)
        x0_uncond = model(x, t_vec, y_null_indices, y_null_offsets)

        x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        t_reshaped = t_vec.view(-1, 1, 1, 1).clamp(min=1e-5)
        v = (x - x0) / t_reshaped
        x = x + dt * v

    return x


if __name__ == "__main__":
    print("=" * 60)
    print("TokenformerDiT Test Suite")
    print("=" * 60)
    device = torch.device("cpu")

    print(f"\n--- Testing with FCDMConditioning ---")
    model = TokenformerDiT(
        in_channels=32, dim=256, depth=12, num_heads=8,
        num_classes=1000, max_tags=64, fcdm_blocks=2,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    y_params = sum(p.numel() for p in model.y_embedder.parameters() if p.requires_grad)
    print(f"  Total: {num_params / 1e6:.2f}M | FCDMConditioning: {y_params / 1e6:.3f}M")

    batch_size, H, W = 2, 8, 8
    x_in = torch.randn(batch_size, 32, H, W, device=device)
    y_indices = torch.randint(0, 100, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)
    t = torch.rand(batch_size, device=device)

    # Test 1: basic forward
    model.eval()
    x0_pred = model(x_in, t, y_indices, y_offsets)
    assert x0_pred.shape == x_in.shape
    print(f"  [OK] Forward: {x0_pred.shape}")

    # Test 2: return_layer_match
    x0_pred, infonce = model(x_in, t, y_indices, y_offsets, return_layer_match=True)
    assert x0_pred.shape == x_in.shape
    assert isinstance(infonce, torch.Tensor)
    print(f"  [OK] Layer match: infonce_loss={infonce.item():.4f}")

    # Test 3: backward
    model.train()
    x0_pred, infonce = model(x_in, t, y_indices, y_offsets, return_layer_match=True)
    loss = F.mse_loss(x0_pred, torch.randn_like(x0_pred)) + 0.2 * infonce
    loss.backward()
    print(f"  [OK] Backward: loss={loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
