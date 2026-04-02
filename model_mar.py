import os
import math
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block


# Helper function for mask
def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
                            src=torch.ones(bsz, seq_len, device=order.device)).bool()
    return masking


def get_2d_sincos_pos_embed(embed_dim, h, w, device, dtype=torch.bfloat16):
    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')

    pos_h = grid_h.reshape(-1)
    pos_w = grid_w.reshape(-1)

    assert embed_dim % 4 == 0
    dim_half = embed_dim // 4

    omega = torch.exp(
        -math.log(10000.0) * torch.arange(0, dim_half, dtype=torch.float32, device=device) / dim_half
    )

    out_h = pos_h[:, None] * omega[None, :]
    out_w = pos_w[:, None] * omega[None, :]

    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)

    # Compute in fp32 for precision, then cast to target dtype
    return torch.cat([emb_h, emb_w], dim=1).unsqueeze(0).to(dtype)


# Flow Matching MLP Components
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
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
        half = dim // 2
        # FIX: keep freqs in same dtype as t, compute in fp32 then cast
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        ).to(dtype=t.dtype)
        args = t[:, None] * freqs[None]  # FIX: removed .float() cast
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # Flow matching uses t in [0, 1]. We scale by 1000 for standard embeddings.
        t_freq = self.timestep_embedding(t * 1000, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SizeEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, h, w, bsz, device):
        # FIX: create tensors in bfloat16 so timestep_embedding stays in bf16
        h_freq = TimestepEmbedder.timestep_embedding(
            torch.tensor([h], dtype=torch.bfloat16, device=device),
            self.frequency_embedding_size
        )
        w_freq = TimestepEmbedder.timestep_embedding(
            torch.tensor([w], dtype=torch.bfloat16, device=device),
            self.frequency_embedding_size
        )
        hw_freq = torch.cat([h_freq, w_freq], dim=-1).expand(bsz, -1)
        return self.mlp(hw_freq)  # FIX: removed redundant .to(torch.bfloat16)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)

        self.fc1 = nn.Linear(channels, channels, bias=True)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(channels, channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y, h_feat, w_feat):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)

        h = self.act(self.fc1(h))

        # DW Conv 3x3
        B, seq_len, C = h.shape
        h = h.transpose(1, 2).view(B, C, h_feat, w_feat)
        h = self.dwconv(h)
        h = h.view(B, C, seq_len).transpose(1, 2)

        h = self.fc2(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks,
                 grad_checkpointing=False):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c, h_feat, w_feat):
        x = x.to(torch.bfloat16)
        x = self.input_proj(x)
        t = self.time_embed(t)
        t = t.unsqueeze(1)
        c = self.cond_embed(c)
        y = t + c
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y, h_feat, w_feat, use_reentrant=False)
        else:
            for block in self.res_blocks:
                x = block(x, y, h_feat, w_feat)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale, h_feat, w_feat):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c, h_feat, w_feat)
        cond_v, uncond_v = torch.split(model_out, len(model_out) // 2, dim=0)
        v = uncond_v + cfg_scale * (cond_v - uncond_v)
        return torch.cat([v, v], dim=0)


class FMLoss(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps,
                 grad_checkpointing=False):
        super().__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )
        self.num_sampling_steps = num_sampling_steps

    def forward(self, target, z, h_feat, w_feat, mask=None):
        noise = torch.randn_like(target)  # inherits target dtype ✅
        B, seq_len, C = target.shape

        # 1. Calculate effective dimension (m) and scaling factor (alpha)
        m = seq_len * C
        n = 32768.0
        alpha = (m / n) ** 0.5

        # FIX: sample t directly in bfloat16 to avoid fp32 upcasting through xt
        t_base = torch.rand((B,), device=target.device, dtype=torch.bfloat16)
        t = (alpha * t_base) / (1.0 + (alpha - 1.0) * t_base)
        t_reshaped = t.view(-1, 1, 1)

        xt = (1 - t_reshaped) * target + t_reshaped * noise
        v_target = noise - target

        v_pred = self.net(xt, t, z, h_feat, w_feat)

        loss = F.mse_loss(v_pred, v_target, reduction='none')
        loss = loss.mean(dim=-1)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        return loss

    def sample(self, z, h_feat, w_feat, temperature=1.0, cfg=1.0):
        B = z.shape[0]
        seq_len = z.shape[1]
        if cfg != 1.0:
            B = B // 2
            noise = torch.randn(B, seq_len, self.in_channels, device=z.device,
                                dtype=torch.bfloat16) * temperature
            noise = torch.cat([noise, noise], dim=0)
            sample_fn = lambda x, t, c: self.net.forward_with_cfg(x, t, c, cfg, h_feat, w_feat)
        else:
            noise = torch.randn(B, seq_len, self.in_channels, device=z.device,
                                dtype=torch.bfloat16) * temperature
            sample_fn = lambda x, t, c: self.net(x, t, c, h_feat, w_feat)

        x = noise
        steps = self.num_sampling_steps
        # FIX: linspace in bfloat16 to avoid fp32 contamination via dt
        ts = torch.linspace(1.0, 0.0, steps + 1, device=z.device, dtype=torch.bfloat16)

        for i in range(steps):
            t_curr = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_curr

            t_vec = torch.full((x.shape[0],), t_curr.item(), device=x.device,
                               dtype=torch.bfloat16)
            v_pred = sample_fn(x, t_vec, z)
            x = x + v_pred * dt

        return x


class MeanPoolingEmbedder(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.embed = nn.EmbeddingBag(num_classes + 1, dim, mode='mean')
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, y_indices: torch.Tensor, y_offsets: torch.Tensor) -> torch.Tensor:
        x = self.embed(y_indices, y_offsets)
        x = x + self.mlp(x)
        return x


class TagProcessor:
    def __init__(self, tags_file):
        with open(tags_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.num_classes = len(self.tags)

    def process_prompts(self, prompts, device, dropout_prob=0.0):
        import random
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


class MARModel(nn.Module):
    def __init__(self, in_channels=128, dim=768, depth=12, num_heads=12, num_classes=12476,
                 latent_size=16, use_checkpoint=False, mask_ratio_min=0.7, label_drop_prob=0.1,
                 buffer_size=64, diffloss_d=3, diffloss_w=512, num_sampling_steps=100,
                 diffusion_batch_mul=4, **kwargs):
        super().__init__()
        self.vae_embed_dim = in_channels
        self.patch_size = 1
        self.token_embed_dim = in_channels * self.patch_size ** 2
        self.grad_checkpointing = use_checkpoint

        self.num_classes = num_classes
        self.y_embedder = MeanPoolingEmbedder(num_classes, dim)
        self.size_embedder = SizeEmbedder(dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, dim))

        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0,
                                                    scale=0.25)

        self.z_proj = nn.Linear(self.token_embed_dim, dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(dim, eps=1e-6)

        self.buffer_size = buffer_size
        self.buffer_pos_embed_encoder = nn.Parameter(torch.zeros(1, buffer_size, dim))
        self.buffer_pos_embed_decoder = nn.Parameter(torch.zeros(1, buffer_size, dim))

        self.encoder_blocks = nn.ModuleList([
            Block(dim, num_heads, 4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(dim)

        self.decoder_embed = nn.Linear(dim, dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.decoder_blocks = nn.ModuleList([
            Block(dim, num_heads, 4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.decoder_norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.buffer_pos_embed_encoder, std=.02)
        torch.nn.init.normal_(self.buffer_pos_embed_decoder, std=.02)

        self.fmloss = FMLoss(
            target_channels=self.token_embed_dim,
            z_channels=dim,
            depth=diffloss_d,
            width=diffloss_w,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=use_checkpoint
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x, h_, w_

    def unpatchify(self, x, h_, w_):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def sample_orders(self, bsz, seq_len, device):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(device).long()
        return orders

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding, h_, w_):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # FIX: match dtype of x so cat doesn't upcast
        x = torch.cat([
            torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device, dtype=x.dtype),
            x
        ], dim=1)
        mask_with_buffer = torch.cat([
            torch.zeros(x.size(0), self.buffer_size, device=x.device),
            mask
        ], dim=1)

        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.device).to(x.dtype)
            class_embedding = (drop_latent_mask * self.fake_latent
                               + (1 - drop_latent_mask) * class_embedding)
        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        pos_embed = get_2d_sincos_pos_embed(embed_dim, h_, w_, x.device, dtype=x.dtype)
        encoder_pos_embed = torch.cat([self.buffer_pos_embed_encoder.to(x.dtype), pos_embed], dim=1)

        x = x + encoder_pos_embed
        x = self.z_proj_ln(x.to(torch.bfloat16))
        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)
        return x

    def forward_mae_decoder(self, x, mask, h_, w_):
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([
            torch.zeros(x.size(0), self.buffer_size, device=x.device),
            mask
        ], dim=1)
        mask_tokens = self.mask_token.repeat(
            mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1
        ).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(
            x.shape[0] * x.shape[1], x.shape[2]
        )

        pos_embed = get_2d_sincos_pos_embed(x.shape[-1], h_, w_, x.device, dtype=x.dtype)
        decoder_pos_embed = torch.cat([self.buffer_pos_embed_decoder.to(x.dtype), pos_embed], dim=1)

        x = x_after_pad + decoder_pos_embed
        x = x.to(torch.bfloat16)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:]
        x = x + pos_embed
        return x

    def forward_loss(self, z, target, mask, h_, w_):
        bsz, seq_len, _ = target.shape
        target = target.repeat(self.diffusion_batch_mul, 1, 1)
        z = z.repeat(self.diffusion_batch_mul, 1, 1)
        mask = mask.repeat(self.diffusion_batch_mul, 1)
        loss = self.fmloss(target=target, z=z, h_feat=h_, w_feat=w_, mask=mask)
        return loss

    def forward(self, imgs, y_indices, y_offsets):
        bsz = imgs.size(0)
        x, h_, w_ = self.patchify(imgs)
        seq_len = h_ * w_

        class_embedding = self.y_embedder(y_indices, y_offsets)
        class_embedding = class_embedding + self.size_embedder(h_, w_, bsz, imgs.device)

        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz, seq_len, imgs.device)
        mask = self.random_masking(x, orders)
        x = self.forward_mae_encoder(x, mask, class_embedding, h_, w_)
        z = self.forward_mae_decoder(x, mask, h_, w_)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask, h_=h_, w_=w_)
        return loss

    def sample_tokens_custom(self, bsz, y_indices, y_offsets, num_classes, h_, w_, num_iter=64,
                             cfg=1.0, cfg_schedule="linear", temperature=1.0):
        seq_len = h_ * w_
        device = y_indices.device

        mask = torch.ones(bsz, seq_len, device=device)  # mask is float for scatter ops
        # FIX: tokens in bfloat16 from the start
        tokens = torch.zeros(bsz, seq_len, self.token_embed_dim, device=device,
                             dtype=torch.bfloat16)
        orders = self.sample_orders(bsz, seq_len, device)

        for step in range(num_iter):
            cur_tokens = tokens.clone()
            class_embedding = self.y_embedder(y_indices, y_offsets)
            class_embedding = class_embedding + self.size_embedder(h_, w_, bsz, tokens.device)

            if cfg != 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                uncond_indices = torch.full((bsz,), num_classes, dtype=torch.long, device=device)
                uncond_offsets = torch.arange(bsz, dtype=torch.long, device=device)
                uncond_embedding = self.y_embedder(uncond_indices, uncond_offsets)
                uncond_embedding = uncond_embedding + self.size_embedder(h_, w_, bsz, tokens.device)

                class_embedding = torch.cat([class_embedding, uncond_embedding], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            x = self.forward_mae_encoder(tokens, mask, class_embedding, h_, w_)
            z = self.forward_mae_decoder(x, mask, h_, w_)

            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(
                torch.Tensor([1]).to(device),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len)
            )
            mask_next = mask_by_order(mask_len[0], orders, bsz, seq_len)

            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next

            if cfg != 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (seq_len - mask_len[0]) / seq_len
            else:
                cfg_iter = cfg

            sampled_token_latent = self.fmloss.sample(z, h_, w_, temperature, cfg_iter)

            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
            sampled_token_latent = sampled_token_latent.to(cur_tokens.dtype)
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = \
                sampled_token_latent[mask_to_pred.nonzero(as_tuple=True)]
            tokens = cur_tokens.clone()

        tokens = self.unpatchify(tokens, h_, w_)
        return tokens


@torch.no_grad()
def sample_mar(model, tag_processor, latent_size, batch_size, prompts, device,
               steps=64, cfg_scale=1.4, **kwargs):
    # FIX: removed unused `noise` parameter; accept **kwargs for compatibility
    model.eval()
    y_indices, y_offsets = tag_processor.process_prompts(prompts[:batch_size], device)

    if isinstance(latent_size, tuple):
        h_, w_ = latent_size
    else:
        h_ = w_ = latent_size

    x = model.sample_tokens_custom(
        bsz=batch_size,
        y_indices=y_indices,
        y_offsets=y_offsets,
        num_classes=tag_processor.num_classes,
        h_=h_,
        w_=w_,
        num_iter=steps,
        cfg=cfg_scale,
        cfg_schedule="linear",
        temperature=1.0
    )
    model.train()
    return x


if __name__ == '__main__':
    print("Testing MARModel instantiation and forward pass...")

    class DummyTagProcessor:
        def __init__(self, num_classes):
            self.num_classes = num_classes

        def process_prompts(self, prompts, device, dropout_prob=0.0):
            bsz = len(prompts)
            indices = torch.randint(0, self.num_classes, (bsz * 5,), device=device)
            offsets = torch.arange(0, bsz * 5, 5, device=device)
            return indices, offsets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    in_channels = 32
    num_classes = 1000
    model = MARModel(
        in_channels=in_channels,
        dim=256,
        depth=4,
        num_heads=4,
        num_classes=num_classes,
        diffloss_d=2,
        diffloss_w=256
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nMARModel initialized.")
    print(f"Total parameters: {num_params / 1e6:.2f} M")

    model.train()
    model.to(torch.bfloat16)
    bsz = 2
    imgs = torch.randn(bsz, in_channels, 16, 16, device=device)
    prompts = ["a beautiful anime girl", "a cool mecha robot"]
    tag_processor = DummyTagProcessor(num_classes)
    y_indices, y_offsets = tag_processor.process_prompts(prompts, device)

    print("\nRunning forward pass (training)...")
    loss = model(imgs, y_indices, y_offsets)
    print(f"Forward pass successful. Loss: {loss.item():.4f}")

    print("\nRunning sample pass (generation)...")
    with torch.no_grad():
        samples = sample_mar(
            model=model,
            tag_processor=tag_processor,
            latent_size=(16, 16),
            batch_size=bsz,
            prompts=prompts,
            device=device,
            steps=10,
            cfg_scale=4.0
        )

    print(f"Sample pass successful.")
    print(f"Generated output shape: {samples.shape}")
    print(f"Expected shape: [{bsz}, {in_channels}, 16, 16]")
    assert samples.shape == (bsz, in_channels, 16, 16), "Output shape mismatch!"
    print("\nAll tests passed successfully! ✅")