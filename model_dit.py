import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint, CheckpointPolicy, create_selective_checkpoint_contexts
from tqdm.auto import tqdm
from torch.utils.flop_counter import FlopCounterMode
import functools
import random

torch._dynamo.config.recompile_limit = 128
aten = torch.ops.aten
compute_intensive_ops = [
    aten.mm.default,
    aten.bmm,
    aten.addmm,
]
torch._functorch.config.activation_memory_budget = 0.5

def policy_fn(ctx, op, *args, **kwargs):
    if op in compute_intensive_ops:
        return CheckpointPolicy.MUST_SAVE
    else:
        return CheckpointPolicy.PREFER_RECOMPUTE


context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)


class TagTransformer(nn.Module):
    """
    Processes variable-length tags separately using a simple 3-layer Transformer blocks stack
    and extracts 4 processed [CLS] tokens as multi-resolution tag embeddings.
    - Stage 1 (encoder/decoder level 1): CLS[0] → dim
    - Stage 2 (encoder/decoder level 2): cat(CLS[0], CLS[1]) → 2*dim
    - Stage 3 (bottleneck):              cat(CLS[0:4]) → 4*dim
    """
    def __init__(self, num_classes: int, dim: int, num_heads: int = 16, depth: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, dim)  # +1 for padding/null class
        self.cls_token = nn.Parameter(torch.zeros(1, 4, dim))
        
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, use_dwconv=False) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
        # Weight initialization
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor = None) -> torch.Tensor:
        # Determine Batch Size
        if y_offsets is not None:
            B = len(y_offsets)
        else:
            B = 1
            y_offsets = torch.zeros(1, dtype=torch.long, device=y_indices.device)
            
        device = y_indices.device
        
        # Calculate individual tag lengths for batch items
        offsets_list = y_offsets.tolist()
        total_len = len(y_indices)
        lengths = []
        for i in range(B):
            start = offsets_list[i]
            end = offsets_list[i+1] if i + 1 < B else total_len
            lengths.append(end - start)
            
        max_len = max(lengths) if lengths else 0
        
        # Pad tag represents the null token/dropout tag at self.embedding.num_embeddings - 1
        padding_val = self.embedding.num_embeddings - 1
        target_len = 32
        padded_tags = torch.full((B, target_len), padding_val, dtype=torch.long, device=device)
        
        for i in range(B):
            start = offsets_list[i]
            end = offsets_list[i+1] if i + 1 < B else total_len
            item_len = end - start
            if item_len > 0:
                copy_len = min(item_len, target_len)
                padded_tags[i, :copy_len] = y_indices[start:start+copy_len]
                
        # Embed tags and prepend learnable CLS tokens
        tag_embeds = self.embedding(padded_tags)  # [B, target_len, dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 4, dim]
        x = torch.cat([cls_tokens, tag_embeds], dim=1)  # [B, 4 + target_len, dim]
        
        c = torch.zeros(B, 1, x.shape[-1], device=device, dtype=x.dtype)
        # Run through tag transformer encoder stack
        qkv_res = None
        for block in self.blocks:
            x, qkv_res = block(x, c, qkv_res=qkv_res)
            
        x = self.norm(x)
        return x[:, :4, :]  # Extract 4 CLS token representations [B, 4, dim]


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
            # Depthwise spatial convolution
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

    def forward(self, x, H=None, W=None, rope=None, num_context_tokens=0, seq_indices=None, src_key_padding_mask=None, qkv_res=None):
        B, N_seq, C = x.shape
        qkv_proj = self.qkv(x)
        if qkv_res is not None:
            qkv_proj = qkv_res + qkv_proj
        qkv = qkv_proj.chunk(3, dim=-1)
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

        # Exclusive Self-Attention (XSA)
        q_f, k_f, v_f = q.float(), k.float(), v.float()
        
        if src_key_padding_mask is not None:
            attn_mask = ~src_key_padding_mask.view(B, 1, 1, N_seq)
            Y = F.scaled_dot_product_attention(q_f, k_f, v_f, attn_mask=attn_mask)
        else:
            Y = F.scaled_dot_product_attention(q_f, k_f, v_f)
            
        Vn = F.normalize(v_f, dim=-1)
        Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn

        x_att = Z.to(q.dtype).transpose(1, 2).reshape(B, N_seq, C)
        return self.proj(x_att), qkv_proj


class DiTBlock(nn.Module):
    """
    Diffusion Transformer (DiT) Block with adaLN-Zero modulation and XSA.
    Supports CLS tokens that participate in attention but bypass the MLP.
    """
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
        
        # Zero-initialize the linear modulation parameters
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
    def forward(self, x, c, H=None, W=None, rope=None, num_cls_tokens=0, src_key_padding_mask=None, qkv_res=None):
        # Handle 2D conditioning tensor: c is [B, dim] -> [B, 1, dim]
        if c.dim() == 2:
            c = c.unsqueeze(1)
            
        modulation = self.adaLN_modulation(c)  # [B, N, 6*dim] or [B, 1, 6*dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        scale_msa = torch.tanh(scale_msa)
        scale_mlp = torch.tanh(scale_mlp)
        # Attention branch — all tokens (CLS + spatial) participate
        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1 * (1 + scale_msa) + shift_msa
        attn_out, qkv_res_out = self.self_attn(x_norm1, H, W, rope, num_context_tokens=num_cls_tokens, src_key_padding_mask=src_key_padding_mask, qkv_res=qkv_res)
        x = x + gate_msa * attn_out

        # MLP branch — only spatial tokens, CLS tokens bypass
        if num_cls_tokens > 0:
            cls = x[:, :num_cls_tokens]
            spatial = x[:, num_cls_tokens:]
            # Slice per-token modulation to spatial positions only
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

        return x, qkv_res_out


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
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


#################################################################################
#                     FCDM Components (ConvNeXt U-Net)                          #
#################################################################################

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm applied to 2D feature maps (channels-first format)."""
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer from ConvNeXt V2."""
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
    ConvNeXt-style block with adaptive LayerNorm-Zero (adaLN-Zero) conditioning.
    Combines depthwise conv and pointwise MLP with adaLN modulation on the channel dimension.
    """
    def __init__(self, dim, mlp_ratio=3.0):
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
        x: (B, C, H, W) spatial features
        c: (B, C) conditioning vector
        """
        h = self.dwconv(x)
        # Compute adaLN parameters: shift, scale, gate
        shift, scale, gate = self.adaLN_modulation(c).unsqueeze(2).unsqueeze(3).chunk(3, dim=1)
        # Apply adaptive LayerNorm-Zero
        h = self.norm(h)
        h = torch.addcmul(shift, h, scale + 1)
        # Pointwise MLP
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        # Apply gate and residual
        h = h * gate
        return x + h


class ConvFinalLayer(nn.Module):
    """Final output layer with adaLN modulation + Conv2d."""
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
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Spatial downsample via Conv2d + PixelUnshuffle(2)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsample via Conv2d + PixelShuffle(2)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.conv(x)


#################################################################################
#                     FCDM-L: Fully Convolutional Diffusion Model               #
#################################################################################

class TokenformerDiT(nn.Module):
    """
    FCDM-L: Fully Convolutional Diffusion Model with TagTransformer conditioning.
    ConvNeXt-based U-Net backbone adapted from 'Reviving ConvNeXt for Efficient
    Convolutional Diffusion Models' (CVPR 2026).

    Conditioning uses TagTransformer's 4 CLS tokens stacked on channel dim:
    - Stage 1 (512d):  CLS[0]
    - Stage 2 (1024d): cat(CLS[0], CLS[1])
    - Stage 3 (2048d): cat(CLS[0:4])
    """

    def __init__(self, in_channels=32, dim=512, depth=None, num_heads=16,
                 num_classes=12477, use_checkpoint=False, **kwargs):
        super().__init__()
        hidden_size = dim
        self.in_channels = in_channels
        self.use_checkpoint = use_checkpoint

        # FCDM-L config
        fcdm_depth = [2, 4, 8, 4, 2]
        mlp_ratio = 3

        # --- Conditioning ---
        # TagTransformer returns 4 CLS tokens, each of dim=hidden_size
        self.y_embedder = TagTransformer(
            num_classes, hidden_size,
            num_heads=min(num_heads, hidden_size // 64),
            depth=3
        )

        # Per-resolution timestep embedders
        self.t_embedder_1 = TimestepEmbedder(hidden_size)       # D
        self.t_embedder_2 = TimestepEmbedder(hidden_size * 2)   # 2D
        self.t_embedder_3 = TimestepEmbedder(hidden_size * 4)   # 4D

        # --- Input embedding ---
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1)

        # --- Encoder Level 1 ---
        self.encoder_level_1 = nn.ModuleList([
            ConvNeXtBlock(hidden_size, mlp_ratio=mlp_ratio)
            for _ in range(fcdm_depth[0])
        ])
        self.down1_2 = Downsample(hidden_size, hidden_size * 2)

        # --- Encoder Level 2 ---
        self.encoder_level_2 = nn.ModuleList([
            ConvNeXtBlock(hidden_size * 2, mlp_ratio=mlp_ratio)
            for _ in range(fcdm_depth[1])
        ])
        self.down2_3 = Downsample(hidden_size * 2, hidden_size * 4)

        # --- Bottleneck ---
        self.latent = nn.ModuleList([
            ConvNeXtBlock(hidden_size * 4, mlp_ratio=mlp_ratio)
            for _ in range(fcdm_depth[2])
        ])

        # --- Decoder Level 2 ---
        self.up3_2 = Upsample(hidden_size * 4, hidden_size * 2)
        self.reduce_chans_2 = nn.Conv2d(hidden_size * 4, hidden_size * 2, kernel_size=1)
        self.decoder_level_2 = nn.ModuleList([
            ConvNeXtBlock(hidden_size * 2, mlp_ratio=mlp_ratio)
            for _ in range(fcdm_depth[3])
        ])

        # --- Decoder Level 1 ---
        self.up2_1 = Upsample(hidden_size * 2, hidden_size)
        self.reduce_chans_1 = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1)
        self.decoder_level_1 = nn.ModuleList([
            ConvNeXtBlock(hidden_size, mlp_ratio=mlp_ratio)
            for _ in range(fcdm_depth[4])
        ])

        # --- Output ---
        self.output_layer = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.final_layer = ConvFinalLayer(hidden_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Input embedding
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Timestep embedder MLPs
        for t_emb in [self.t_embedder_1, self.t_embedder_2, self.t_embedder_3]:
            nn.init.normal_(t_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(t_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation in all ConvNeXt blocks
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

    def forward(self, x_in, t, y_indices, y_offsets=None, return_features=False,
                return_layer_match=False, **kwargs):
        """
        Forward pass of FCDM-L.
        x_in: (B, C, H, W) spatial input (latent representation)
        t:    (B,) diffusion timesteps
        y_indices: flat tag index tensor
        y_offsets: per-sample offsets into y_indices
        """
        B, C, H, W = x_in.shape

        # --- Input embedding ---
        x_emb = self.x_embedder(x_in)

        # --- Tag conditioning: (B, 4, dim) ---
        y_cls = self.y_embedder(y_indices, y_offsets)

        # --- Per-resolution conditioning ---
        # Stage 1: 1 CLS token → dim
        t1 = self.t_embedder_1(t)                                  # (B, D)
        c1 = t1 + y_cls[:, 0, :]                                   # (B, D)

        # Stage 2: 2 CLS tokens stacked → 2*dim
        t2 = self.t_embedder_2(t)                                  # (B, 2D)
        y2 = torch.cat([y_cls[:, 0, :], y_cls[:, 1, :]], dim=-1)   # (B, 2D)
        c2 = t2 + y2

        # Stage 3 (bottleneck): 4 CLS tokens stacked → 4*dim
        t3 = self.t_embedder_3(t)                                  # (B, 4D)
        y3 = y_cls.reshape(B, -1)                                   # (B, 4D)
        c3 = t3 + y3

        # --- Encoder Level 1 ---
        out_enc_level1 = x_emb
        for block in self.encoder_level_1:
            if self.use_checkpoint and self.training:
                out_enc_level1 = torch.utils.checkpoint.checkpoint(
                    block, out_enc_level1, c1, use_reentrant=False)
            else:
                out_enc_level1 = block(out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # --- Encoder Level 2 ---
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            if self.use_checkpoint and self.training:
                out_enc_level2 = torch.utils.checkpoint.checkpoint(
                    block, out_enc_level2, c2, use_reentrant=False)
            else:
                out_enc_level2 = block(out_enc_level2, c2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # --- Bottleneck ---
        latent = inp_enc_level3
        for block in self.latent:
            if self.use_checkpoint and self.training:
                latent = torch.utils.checkpoint.checkpoint(
                    block, latent, c3, use_reentrant=False)
            else:
                latent = block(latent, c3)

        # --- Decoder Level 2 ---
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        out_dec_level2 = self.reduce_chans_2(inp_dec_level2)
        for block in self.decoder_level_2:
            if self.use_checkpoint and self.training:
                out_dec_level2 = torch.utils.checkpoint.checkpoint(
                    block, out_dec_level2, c2, use_reentrant=False)
            else:
                out_dec_level2 = block(out_dec_level2, c2)

        # --- Decoder Level 1 ---
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.reduce_chans_1(inp_dec_level1)
        for block in self.decoder_level_1:
            if self.use_checkpoint and self.training:
                out_dec_level1 = torch.utils.checkpoint.checkpoint(
                    block, out_dec_level1, c1, use_reentrant=False)
            else:
                out_dec_level1 = block(out_dec_level1, c1)

        # --- Output ---
        x = self.output_layer(out_dec_level1)
        v_pred = self.final_layer(x, c1)

        # --- Interface compatibility ---
        res = [v_pred]
        if return_features:
            res.append(None)
        if return_layer_match:
            res.append(torch.tensor(0.0, device=v_pred.device))

        if len(res) == 1:
            return res[0]
        return tuple(res)


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
                sampler_type="euler", p_percentile=0.4, alpha=2.0):
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
    #x = x.to(torch.bfloat16)
    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)
    null_prompts = [""] * batch_size
    y_null_indices, y_null_offsets = tag_processor.process_prompts(null_prompts, device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in tqdm(range(steps)):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_vec = torch.full((batch_size,), t_curr.item(), device=device, dtype=x.dtype)

        if sampler_type != "euler":
            raise NotImplementedError(f"Sampler type '{sampler_type}' is not supported anymore because log-variance has been removed. Use 'euler' instead.")

        v_cond = model(x, t_vec, y_indices, y_offsets)
        v_uncond = model(x, t_vec, y_null_indices, y_null_offsets)

        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        x = x + dt * v

    model.train()
    return x


if __name__ == "__main__":
    print("Initializing FCDM-L on CPU...")
    device = torch.device("cpu")

    model = TokenformerDiT(
        in_channels=32,
        dim=512,
        num_heads=8,
        num_classes=100,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params / 1e6:.2f} M")

    batch_size = 2
    H, W = 32, 32  # f8 latent at 256px

    x_in = torch.randn(batch_size, 32, H, W, device=device)
    y_indices = torch.randint(0, 100, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)

    # --- Test 1: Forward pass ---
    print("\n[Test 1] Forward pass...")
    t_global = torch.rand(batch_size, device=device)
    model.eval()
    v_pred, match_loss = model(x_in, t_global, y_indices, y_offsets, return_layer_match=True)
    print(f"  v_pred: {v_pred.shape}, match_loss: {match_loss.item():.6f}")
    assert v_pred.shape == x_in.shape, f"Shape mismatch: {v_pred.shape} vs {x_in.shape}"

    # --- Test 2: Training backward pass ---
    print("[Test 2] Training backward pass...")
    model.train()
    v_pred3, match_loss3 = model(x_in, t_global, y_indices, y_offsets, return_layer_match=True)
    loss = F.mse_loss(v_pred3, torch.randn_like(v_pred3)) + 0.2 * match_loss3
    loss.backward()
    print(f"  loss: {loss.item():.6f}")

    print("\n[SUCCESS] All tests passed!")