import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
import random

torch._dynamo.config.recompile_limit = 128


# =============================================================================
# Utilities
# =============================================================================
class LayerNorm2d(nn.LayerNorm):
    """Channels-last LayerNorm applied in NCHW layout."""
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def modulate(x, shift, scale):
    """Apply adaptive modulation: x * (1 + scale) + shift with spatial broadcast."""
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


# =============================================================================
# Embedding Layers for Timesteps and Tags
# =============================================================================
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal embedding + MLP."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FCDMConditioning(nn.Module):
    """Tag conditioning: EmbeddingBag (mean pooling) followed by an MLP."""
    def __init__(self, num_classes: int, dim: int, max_tags: int = 64, **kwargs):
        super().__init__()
        self.dim = dim
        self.max_tags = max_tags
        self.num_classes = num_classes
        self.padding_idx = num_classes

        self.embedding_bag = nn.EmbeddingBag(
            num_classes + 1, dim, mode='mean', padding_idx=self.padding_idx,
        )
        nn.init.normal_(self.embedding_bag.weight, std=0.02)
        with torch.no_grad():
            self.embedding_bag.weight[self.padding_idx].zero_()

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor = None) -> torch.Tensor:
        if y_offsets is None:
            # Single-sample inference
            indices = y_indices[:self.max_tags]
            offsets = torch.zeros(1, dtype=torch.long, device=y_indices.device)
        else:
            # Clamp per-sample lengths to max_tags and rebuild flat indices
            B = len(y_offsets)
            total_len = len(y_indices)
            next_off = torch.cat([y_offsets[1:], total_len.unsqueeze(0) if isinstance(total_len, torch.Tensor) else torch.tensor([total_len], device=y_offsets.device)])
            lengths = (next_off - y_offsets).clamp(max=self.max_tags)

            chunks = []
            new_offsets = []
            running = 0
            for i in range(B):
                clen = lengths[i].item()
                if clen > 0:
                    chunks.append(y_indices[y_offsets[i]:y_offsets[i] + clen])
                new_offsets.append(running)
                running += clen

            indices = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.long, device=y_indices.device)
            offsets = torch.tensor(new_offsets, dtype=torch.long, device=y_indices.device)

        pooled = self.embedding_bag(indices, offsets)  # [B, dim]
        return self.mlp(pooled)


# =============================================================================
# Core FCDM Components
# =============================================================================
class GRN(nn.Module):
    """GRN (Global Response Normalization) layer in NCHW format."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block with adaLN-Zero conditioning (all NCHW).
    DWConv → Norm+Modulate → Expand(1x1) → GELU → GRN → Reduce(1x1) → gate → residual.
    """
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, affine=False, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.act = nn.GELU()
        self.grn = GRN(int(dim * mlp_ratio))
        self.pwconv2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, C, H, W) feature map
        c: (B, C) conditioning vector
        """
        h = self.dwconv(x)
        # adaLN-Zero: compute shift, scale, gate from conditioning
        shift, scale, gate = self.adaLN_modulation(c).unsqueeze(2).unsqueeze(3).chunk(3, dim=1)
        h = self.norm(h)
        h = torch.addcmul(shift, h, scale + 1)
        # Pointwise MLP
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        # Gate and residual
        h = h * gate
        return x + h


class AttentionBlock(nn.Module):
    """
    Global spatial self-attention with adaLN-Zero conditioning (NCHW).

    Resolution-agnostic (fully convolutional philosophy): attention runs over
    the H*W spatial tokens of the feature map. Uses SDPA so FlashAttention /
    memory-efficient kernels are used on GPU. The residual gate is
    zero-initialized, so the block starts as an identity mapping.
    """
    def __init__(self, dim, num_heads=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads or max(1, dim // 64)
        assert dim % self.num_heads == 0, (
            f"dim {dim} not divisible by num_heads {self.num_heads}")
        self.head_dim = dim // self.num_heads

        self.norm = LayerNorm2d(dim, affine=False, eps=1e-6)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True)
        )

    def forward(self, x, c):
        B, C, H, W = x.shape
        shift, scale, gate = self.adaLN_modulation(c).unsqueeze(2).unsqueeze(3).chunk(3, dim=1)
        h = torch.addcmul(shift, self.norm(x), scale + 1)

        q, k, v = self.qkv(h).chunk(3, dim=1)

        def to_heads(t):  # (B, C, H, W) -> (B, heads, H*W, head_dim)
            return t.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        h = F.scaled_dot_product_attention(to_heads(q), to_heads(k), to_heads(v))
        h = h.transpose(2, 3).reshape(B, C, H, W)

        return x + gate * self.proj(h)


# =============================================================================
# LFQ Memory Block (Non-Causal) — 1D Sequence (B, D, C) + 2D Wrapper (B, C, H, W)
# =============================================================================
class LFQMemoryBlockNonCausal(nn.Module):
    def __init__(self, c_in=1024, c_mem=512, num_latents=8, k_bits=18):
        """
        Args:
            c_in: Input feature dimension (C)
            c_mem: Memory embedding dimension
            num_latents: Number of parallel codebooks (8)
            k_bits: Bits per code (codebook size = 2^k_bits, e.g., 2^18)
        """
        super().__init__()
        self.num_latents = num_latents
        self.k_bits = k_bits
        self.c_mem = c_mem
        self.vocab_size = 2 ** k_bits

        # 1. Projection to 8 latents of K bits
        self.in_proj = nn.Linear(c_in, num_latents * k_bits, bias=False)

        # 2. Memory projections (replaces 2^K embedding tables)
        # Old: 8 x [2^K, c_mem] tables (~1B params for K=18). Gradient to
        # in_proj was blocked by F.embedding (discrete idx).
        # New: 8 x Linear(K -> c_mem) (~74K params for K=18, c_mem=512).
        # Fully differentiable via STE bits.
        self.mem_projs = nn.ModuleList([
            nn.Linear(k_bits, c_mem, bias=False) for _ in range(num_latents)
        ])

        # 3. Gating projections (Key and Value)
        self.k_proj = nn.Linear(c_mem, c_in, bias=False)
        self.v_proj = nn.Linear(c_mem, c_in, bias=False)

        # 4. Normalizations
        self.q_norm = nn.RMSNorm(c_in)
        self.k_norm = nn.RMSNorm(c_in)
        self.v_norm = nn.RMSNorm(c_in)

        # 5. Point-wise Linear Output Projection
        self.out_proj = nn.Linear(c_in, c_in, bias=False)

        # Kept for compatibility/debug (not used in linear mode)
        self.vocab_size = 2 ** k_bits
        self.register_buffer("basis", 2 ** torch.arange(k_bits, dtype=torch.long))

    def _quantize_bits(self, z):
        # STE for LFQ bits: forward hard 0/1, backward sigmoid-STE
        # z: [..., K]  continuous logits
        # 1) sign STE: z_ste = z + (sign(z)-z).detach() -> gradient identity, forward sign(z)
        # 2) bits = (z_ste+1)/2 in [0,1] (0 for -1, 0.5 for 0, 1 for 1)
        # This makes bits differentiable w.r.t z (dbits/dz = 0.5)
        z_sign = torch.sign(z)
        z_ste = z + (z_sign - z).detach()
        bits = (z_ste + 1) * 0.5
        bits = bits.clamp(0, 1)
        return bits

    def _quantize(self, z):
        # Legacy API kept for compatibility: returns indices (not used in linear path)
        z_sign = torch.sign(z)
        z_quantized = z + (z_sign - z).detach()
        bits = (z_quantized > 0).long()
        indices = (bits * self.basis).sum(dim=-1)
        return indices

    def forward(self, x):
        """
        Input x: [B, D, C]
        """
        B, D, C = x.shape

        # Step 1: Project & split into 8 latents
        z_raw = self.in_proj(x)  # [B, D, 8*K]
        z = z_raw.view(B, D, self.num_latents, self.k_bits).permute(2, 0, 1, 3)  # [8, B, D, K]

        # Step 2: Quantize to bits (STE) and project to memory dim via Linear
        retrieved_mem = []
        for i in range(self.num_latents):
            bits = self._quantize_bits(z[i])          # [B, D, K]  differentiable
            emb_i = self.mem_projs[i](bits)            # [B, D, c_mem]  Linear(K -> c_mem)
            retrieved_mem.append(emb_i)

        # Step 3: Sum the 8 memory projections
        e = torch.stack(retrieved_mem, dim=0).sum(dim=0)  # [B, D, c_mem]

        # Step 4: Context-aware Gating
        q = self.q_norm(x)
        k = self.k_norm(self.k_proj(e))
        v = self.v_proj(e)

        alpha = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / (C ** 0.5))
        v_gated = alpha * v  # [B, D, C]

        # Step 5: Point-wise Non-linearity & Output Projection
        y = self.out_proj(F.silu(self.v_norm(v_gated)))

        # Step 6: Residual Addition
        return x + y


class LFQMemoryBlock2D(nn.Module):
    """
    NCHW wrapper around LFQMemoryBlockNonCausal.
    Converts (B, C, H, W) -> (B, H*W, C) -> LFQ -> (B, C, H, W).
    Use this for FCDM which is fully convolutional / NCHW.
    """
    def __init__(self, c_in, c_mem=512, num_latents=8, k_bits=18):
        super().__init__()
        self.block = LFQMemoryBlockNonCausal(
            c_in=c_in, c_mem=c_mem, num_latents=num_latents, k_bits=k_bits
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, D, C]
        y_seq = self.block(x_seq)  # [B, D, C]
        y = y_seq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return y


class TopKMemoryBlock(nn.Module):
    def __init__(self, c_in=1024, c_mem=512, num_entries=4096, top_k=8, query_dim=64):
        super().__init__()
        self.num_entries = num_entries
        self.top_k = top_k
        self.c_mem = c_mem
        self.query_dim = query_dim
        self.q_proj = nn.Linear(c_in, query_dim, bias=False)
        self.memory_keys = nn.Parameter(torch.randn(num_entries, query_dim) * (query_dim ** -0.5))
        self.memory_values = nn.Parameter(torch.randn(num_entries, c_mem) * 0.02)
        self.k_proj = nn.Linear(c_mem, c_in, bias=False)
        self.v_proj = nn.Linear(c_mem, c_in, bias=False)
        self.out_proj = nn.Linear(c_in, c_in, bias=False)
        self.q_norm = nn.RMSNorm(c_in)
        self.k_norm = nn.RMSNorm(c_in)
        self.v_norm = nn.RMSNorm(c_in)

    def forward(self, x):
        B, D, C = x.shape
        q = self.q_proj(x)  # [B,D,query_dim]
        scores = torch.matmul(q, self.memory_keys.t()) / (self.query_dim ** 0.5)  # [B,D,num_entries]
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)  # [B,D,top_k]
        topk_weights = F.softmax(topk_scores, dim=-1)
        retrieved_values = F.embedding(topk_indices, self.memory_values)  # [B,D,top_k,c_mem]
        e = (retrieved_values * topk_weights.unsqueeze(-1)).sum(dim=-2)  # [B,D,c_mem]
        ctx_q = self.q_norm(x)
        ctx_k = self.k_norm(self.k_proj(e))
        ctx_v = self.v_proj(e)
        alpha = torch.sigmoid((ctx_q * ctx_k).sum(dim=-1, keepdim=True) / (C ** 0.5))
        v_gated = alpha * ctx_v
        y = self.out_proj(F.silu(self.v_norm(v_gated)))
        return x + y, topk_indices


class TopKMemoryBlock2D(nn.Module):
    """
    NCHW wrapper around TopKMemoryBlock.
    Converts (B, C, H, W) -> (B, H*W, C) -> TopK -> (B, C, H, W).
    Returns only the residual output (indices are internal for debugging).
    """
    def __init__(self, c_in, c_mem=512, num_entries=4096, top_k=8, query_dim=64):
        super().__init__()
        self.block = TopKMemoryBlock(c_in=c_in, c_mem=c_mem, num_entries=num_entries, top_k=top_k, query_dim=query_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,D,C]
        y_seq, _ = self.block(x_seq)  # [B,D,C]
        y = y_seq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return y


class ConvFinalLayer(nn.Module):
    """Conv-style final layer with adaLN modulation (shift + scale only, no gate)."""
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Spatial downsample via Conv2d + PixelUnshuffle (2x)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsample via Conv2d + PixelShuffle (2x)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.conv(x)


# =============================================================================
# FCDM U-Net Backbone
# =============================================================================
def _stage_blocks(dim, num_blocks, mlp_ratio, attn_every=0):
    """ConvNeXt blocks with an AttentionBlock after every `attn_every` blocks."""
    blocks = []
    for i in range(num_blocks):
        blocks.append(ConvNeXtBlock(dim, mlp_ratio=mlp_ratio))
        if attn_every and (i + 1) % attn_every == 0:
            blocks.append(AttentionBlock(dim))
    return nn.ModuleList(blocks)


class FCDM(nn.Module):
    """
    Fully Convolutional Diffusion Model (FCDM).
    ConvNeXt-based U-Net with adaLN-Zero conditioning, PixelShuffle up/down,
    concat skip connections, per-resolution conditioning, and optional
    attention in the final (decoder) stages.
    """

    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 128,
        depth=[2, 4, 8, 4, 2],
        num_classes: int = 1000,
        use_checkpoint: bool = False,
        max_tags: int = 64,
        mlp_ratio: float = 3.0,
        learn_sigma: bool = False,
        attn_every: int = 3,
        # LFQ Memory at end of Block 2 (encoder_level_2) - legacy
        use_lfq_memory: bool = False,
        lfq_c_mem: int = 512,
        lfq_num_latents: int = 8,
        lfq_k_bits: int = 18,
        # Top-K Memory at end of Block 2 (encoder_level_2) - current
        use_topk_memory: bool = False,
        topk_c_mem: int = 512,
        topk_num_entries: int = 4096,
        topk_top_k: int = 8,
        topk_query_dim: int = 64,
        **kwargs,  # absorb old params: expansion_ratio, kernel_size, cond_dim, fcdm_blocks
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.dim = dim
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint

        # Normalize depth: int → [d, 2d, 4d, 2d, d], list of 5 used directly
        if isinstance(depth, int):
            depth = [depth, depth * 2, depth * 4, depth * 2, depth]
        assert len(depth) == 5, f"depth must be int or list of 5, got {depth}"

        # ---- Input projection ------------------------------------------------
        self.x_embedder = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)

        # ---- Per-resolution timestep embedders --------------------------------
        self.t_embedder_1 = TimestepEmbedder(dim)
        self.t_embedder_2 = TimestepEmbedder(dim * 2)
        self.t_embedder_3 = TimestepEmbedder(dim * 4)

        # ---- Tag conditioning: shared EmbeddingBag + per-resolution projections -
        self.y_embedder = FCDMConditioning(
            num_classes=num_classes, dim=dim, max_tags=max_tags,
        )
        self.y_proj_2 = nn.Linear(dim, dim * 2)
        self.y_proj_3 = nn.Linear(dim, dim * 4)

        # ---- Encoder level 1 -------------------------------------------------
        self.encoder_level_1 = nn.ModuleList([
            ConvNeXtBlock(dim, mlp_ratio=mlp_ratio) for _ in range(depth[0])
        ])
        self.down1_2 = Downsample(dim, dim * 2)

        # ---- Encoder level 2 -------------------------------------------------
        self.encoder_level_2 = nn.ModuleList([
            ConvNeXtBlock(dim * 2, mlp_ratio=mlp_ratio) for _ in range(depth[1])
        ])
        # Memory Block at end of encoder_level_2 (Block 2)
        # Supports LFQ (legacy, Linear) and Top-K (current, sparse).
        # Only one is active; TopK takes precedence if both enabled.
        self.lfq_memory = None
        self.topk_memory = None
        if use_topk_memory:
            self.topk_memory = TopKMemoryBlock2D(
                c_in=dim * 2, c_mem=topk_c_mem,
                num_entries=topk_num_entries, top_k=topk_top_k, query_dim=topk_query_dim
            )
        elif use_lfq_memory:
            self.lfq_memory = LFQMemoryBlock2D(
                c_in=dim * 2, c_mem=lfq_c_mem,
                num_latents=lfq_num_latents, k_bits=lfq_k_bits
            )
        self.down2_3 = Downsample(dim * 2, dim * 4)

        # ---- Bottleneck (latent) ---------------------------------------------
        self.latent = nn.ModuleList([
            ConvNeXtBlock(dim * 4, mlp_ratio=mlp_ratio) for _ in range(depth[2])
        ])

        # ---- Decoder level 2 -------------------------------------------------
        self.up3_2 = Upsample(dim * 4, dim * 2)
        self.reduce_chans_2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.decoder_level_2 = _stage_blocks(dim * 2, depth[3], mlp_ratio, attn_every)

        # ---- Decoder level 1 -------------------------------------------------
        self.up2_1 = Upsample(dim * 2, dim)
        self.reduce_chans_1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder_level_1 = _stage_blocks(dim, depth[4], mlp_ratio, attn_every)

        # ---- Output -----------------------------------------------------------
        self.output_layer = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.final_layer = ConvFinalLayer(dim, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Xavier init for all Linear and Conv2d
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Input projection (re-init as flat matrix)
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Tag embedding (re-init after _basic_init overwrote it)
        nn.init.normal_(self.y_embedder.embedding_bag.weight, std=0.02)
        with torch.no_grad():
            self.y_embedder.embedding_bag.weight[self.y_embedder.padding_idx].zero_()

        # Timestep embedding MLPs
        for t_emb in [self.t_embedder_1, self.t_embedder_2, self.t_embedder_3]:
            nn.init.normal_(t_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(t_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in all blocks
        all_blocks = (
            list(self.encoder_level_1) + list(self.encoder_level_2) +
            list(self.latent) +
            list(self.decoder_level_2) + list(self.decoder_level_1)
        )
        for block in all_blocks:
            if hasattr(block, 'adaLN_modulation'):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        # Memory blocks: zero out_proj so block starts as identity residual
        if getattr(self, 'lfq_memory', None) is not None:
            nn.init.constant_(self.lfq_memory.block.out_proj.weight, 0)
        if getattr(self, 'topk_memory', None) is not None:
            nn.init.constant_(self.topk_memory.block.out_proj.weight, 0)
            # Re-init keys/values with small std after _basic_init overwrote them
            nn.init.normal_(self.topk_memory.block.memory_keys, std=self.topk_memory.block.query_dim ** -0.5)
            nn.init.normal_(self.topk_memory.block.memory_values, std=0.02)

    def _block_forward(self, block, x, c):
        if self.use_checkpoint and self.training:
            return checkpoint(block, x, c, use_reentrant=False)
        return block(x, c)

    def _lfq_forward(self, x):
        if self.lfq_memory is None:
            return x
        if self.use_checkpoint and self.training:
            return checkpoint(self.lfq_memory, x, use_reentrant=False)
        return self.lfq_memory(x)

    def _topk_forward(self, x):
        if self.topk_memory is None:
            return x
        if self.use_checkpoint and self.training:
            return checkpoint(self.topk_memory, x, use_reentrant=False)
        return self.topk_memory(x)

    def forward(
        self,
        x_in,
        t,
        y_indices,
        y_offsets=None,
        r=None,
        return_features=False,
        return_layer_match=False,
        **kwargs,
    ):
        """
        Forward pass of FCDM.
        x_in: (B, C, H, W) input (images or latent representations)
        t: (B,) diffusion timesteps in [0, 1]
        y_indices: flat 1-D tag indices
        y_offsets: per-sample offsets into y_indices
        r: (B,) optional interval end time for EMF-style mean-flow models.
           When given, the model predicts the flow over [t, r] and is
           conditioned on (r - t) through the same timestep embedders.
        """
        # Project input
        x = self.x_embedder(x_in)

        # Per-resolution conditioning: time + tags
        t_scaled = t * 1000.0
        y_base = self.y_embedder(y_indices, y_offsets)  # [B, dim]

        if r is not None:
            dr_scaled = (r - t) * 1000.0
            c1 = self.t_embedder_1(t_scaled) + self.t_embedder_1(dr_scaled) + y_base
            c2 = self.t_embedder_2(t_scaled) + self.t_embedder_2(dr_scaled) + self.y_proj_2(y_base)
            c3 = self.t_embedder_3(t_scaled) + self.t_embedder_3(dr_scaled) + self.y_proj_3(y_base)
        else:
            c1 = self.t_embedder_1(t_scaled) + y_base
            c2 = self.t_embedder_2(t_scaled) + self.y_proj_2(y_base)
            c3 = self.t_embedder_3(t_scaled) + self.y_proj_3(y_base)

        # Encoder level 1
        out_enc_level1 = x
        for block in self.encoder_level_1:
            out_enc_level1 = self._block_forward(block, out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # Encoder level 2
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            out_enc_level2 = self._block_forward(block, out_enc_level2, c2)
        # Memory Block at end of Block 2 (non-causal, NCHW-friendly)
        # TopK takes precedence over LFQ
        if self.topk_memory is not None:
            out_enc_level2 = self._topk_forward(out_enc_level2)
        elif self.lfq_memory is not None:
            out_enc_level2 = self._lfq_forward(out_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # Bottleneck
        latent = inp_enc_level3
        for block in self.latent:
            latent = self._block_forward(block, latent, c3)

        # Decoder level 2 (concat skip)
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = inp_dec_level2[:, :, :out_enc_level2.shape[2], :out_enc_level2.shape[3]]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        out_dec_level2 = self.reduce_chans_2(inp_dec_level2)
        for block in self.decoder_level_2:
            out_dec_level2 = self._block_forward(block, out_dec_level2, c2)

        # Decoder level 1 (concat skip)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = inp_dec_level1[:, :, :out_enc_level1.shape[2], :out_enc_level1.shape[3]]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.reduce_chans_1(inp_dec_level1)
        for block in self.decoder_level_1:
            out_dec_level1 = self._block_forward(block, out_dec_level1, c1)

        # Output
        x = self.output_layer(out_dec_level1)
        x0_pred = self.final_layer(x, c1)

        # Compatibility returns
        infonce_loss = torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)

        if return_features:
            return x0_pred, x
        if return_layer_match:
            return x0_pred, infonce_loss
        return x0_pred


# =============================================================================
# FCDM Model Configs
# =============================================================================
def FCDM_S(**kwargs):
    return FCDM(dim=128, depth=[2, 4, 8, 4, 2], **kwargs)

def FCDM_B(**kwargs):
    return FCDM(dim=256, depth=[2, 4, 8, 4, 2], **kwargs)

def FCDM_L(**kwargs):
    return FCDM(dim=512, depth=[2, 4, 8, 4, 2], **kwargs)

def FCDM_XL(**kwargs):
    return FCDM(dim=512, depth=[3, 6, 12, 6, 3], **kwargs)

FCDM_models = {
    'FCDM-S': FCDM_S,
    'FCDM-B': FCDM_B,
    'FCDM-L': FCDM_L,
    'FCDM-XL': FCDM_XL,
}


# =============================================================================
# Tag Processor
# =============================================================================
class TagProcessor:
    def __init__(self, tags_file):
        with open(tags_file, "r", encoding="utf-8") as f:
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


# =============================================================================
# Sampling (Rectified-Flow / FM sampler)
# =============================================================================
@torch.no_grad()
def sample_flow(model, tag_processor, latent_size, batch_size, prompts, device,
                steps=250, guidance_scale=1.5, noise=None):
    in_channels = model.in_channels if hasattr(model, "in_channels") else (
        model.module.in_channels if hasattr(model, "module") else 32)
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
    for i in tqdm(range(steps), desc="Sampling", disable=device.type == "cpu"):
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


# =============================================================================
# Quick self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FCDM Test Suite")
    print("=" * 60)
    device = torch.device("cpu")

    model = FCDM(
        in_channels=4,
        dim=128,
        depth=[2, 4, 8, 4, 2],
        num_classes=1000,
        max_tags=64,
        mlp_ratio=3.0,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    y_params = sum(p.numel() for p in model.y_embedder.parameters() if p.requires_grad)
    print(f"  Total params: {num_params / 1e6:.2f}M | Conditioning: {y_params / 1e6:.3f}M")

    batch_size, H, W = 2, 32, 32
    x_in = torch.randn(batch_size, 4, H, W, device=device)
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
    print(f"  [OK] Layer match placeholder: infonce_loss={infonce.item():.4f}")

    # Test 3: backward
    model.train()
    x0_pred, infonce = model(x_in, t, y_indices, y_offsets, return_layer_match=True)
    loss = F.mse_loss(x0_pred, torch.randn_like(x0_pred)) + 0.2 * infonce
    loss.backward()
    print(f"  [OK] Backward: loss={loss.item():.4f}")

    # Test 4: learn_sigma
    model_sigma = FCDM(in_channels=4, dim=64, depth=[1, 2, 4, 2, 1], num_classes=100, learn_sigma=True).to(device)
    out = model_sigma(x_in, t, y_indices, y_offsets)
    assert out.shape == (batch_size, 8, H, W), f"Expected 8 channels with learn_sigma, got {out.shape}"
    print(f"  [OK] learn_sigma: {out.shape}")

    # Test 5: depth as int (backward compat)
    model_int = FCDM(in_channels=4, dim=64, depth=2, num_classes=100).to(device)
    out = model_int(x_in, t, y_indices, y_offsets)
    assert out.shape == x_in.shape
    print(f"  [OK] depth=int: {out.shape}")

    # Test 6: interval conditioning (EMF-style r argument)
    r = (t + torch.rand(batch_size, device=device) * (1.0 - t)).clamp(max=1.0)
    out_r = model(x_in, t, y_indices, y_offsets, r=r)
    assert out_r.shape == x_in.shape
    loss = F.mse_loss(out_r, torch.randn_like(out_r))
    loss.backward()
    print(f"  [OK] interval r: {out_r.shape}")

    # Test 7: decoder attention + rectangular input (fully convolutional)
    n_attn = sum(isinstance(b, AttentionBlock) for b in
                 list(model.decoder_level_1) + list(model.decoder_level_2))
    assert n_attn > 0, "expected AttentionBlocks in decoder stages"
    x_rect = torch.randn(2, 4, 32, 48, device=device)
    out_rect = model(x_rect, t, y_indices, y_offsets)
    assert out_rect.shape == x_rect.shape
    print(f"  [OK] decoder attention ({n_attn} blocks), rect input: {tuple(out_rect.shape)}")

    # Test 8: attention disabled (attn_every=0)
    model_noattn = FCDM(in_channels=4, dim=64, depth=2, num_classes=100, attn_every=0).to(device)
    out_na = model_noattn(x_in, t, y_indices, y_offsets)
    assert out_na.shape == x_in.shape
    print(f"  [OK] attn_every=0: {out_na.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)