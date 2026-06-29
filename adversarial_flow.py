"""
Adversarial Flow Models components adapted to nanoWaifu's latent DiT.

The losses and gradient normalization follow the official implementation:
https://github.com/ByteDance-Seed/Adversarial-Flow-Models
See THIRD_PARTY_LICENSES/Adversarial-Flow-Models.txt.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model_dit import DiTBlock, GGRoPE2d, TagTransformer, TimestepEmbedder


def interpolate_flow(data, noise, timesteps):
    timesteps = timesteps.reshape(-1, 1, 1, 1).to(dtype=data.dtype)
    return (1.0 - timesteps) * data + timesteps * noise


def sample_afm_timesteps(batch_size, steps, device):
    """Sample one of the designated AFM transitions t -> t - 1 / steps."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("adversarial.steps must be a positive integer")

    indices = torch.randint(0, steps, (batch_size,), device=device)
    timesteps_src = 1.0 - indices.float() / steps
    timesteps_tgt = timesteps_src - 1.0 / steps
    return timesteps_src, timesteps_tgt


def cosine_decay(step, start, end, decay_steps):
    if decay_steps <= 0:
        return float(end)
    progress = min(max(step, 0) / decay_steps, 1.0)
    amount = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(end + (start - end) * amount)


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class GradientNormalization(nn.Module):
    """Normalize the discriminator gradient entering the generator."""

    def __init__(self, ema_decay=0.9, eps=1e-8, target_scale=1.0):
        super().__init__()
        self.ema_decay = ema_decay
        self.eps = eps
        self.target_scale = target_scale
        self.register_buffer("square_avg", torch.tensor(0.0))

    def forward(self, inputs):
        return _GradientNormalizationFn.apply(
            inputs,
            self.square_avg,
            self.ema_decay,
            self.eps,
            self.target_scale,
        )


class _GradientNormalizationFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, square_avg, ema_decay, eps, target_scale):
        ctx.square_avg = square_avg
        ctx.ema_decay = ema_decay
        ctx.eps = eps
        ctx.target_scale = target_scale
        return inputs.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_sq_sum = (
            grad_output.float().square().sum() * grad_output.numel()
        )
        if dist.is_initialized():
            dist.all_reduce(grad_sq_sum, op=dist.ReduceOp.AVG)

        ctx.square_avg.lerp_(grad_sq_sum, 1.0 - ctx.ema_decay)
        scale = ctx.square_avg.sqrt() + ctx.eps
        multiplier = (ctx.target_scale / scale).to(grad_output.dtype)
        grad_output = grad_output * multiplier
        return grad_output, None, None, None, None


class AdversarialFlowDiscriminator(nn.Module):
    """Time- and tag-conditioned latent discriminator with a CLS readout."""

    def __init__(
        self,
        in_channels,
        dim,
        depth,
        num_heads,
        num_classes,
        use_checkpoint=False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Discriminator dim must be divisible by num_heads")

        self.in_channels = in_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.patch_embed = nn.Linear(in_channels, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = TagTransformer(
            num_classes,
            dim,
            num_heads=num_heads,
            depth=min(3, depth),
        )
        self.rope = GGRoPE2d(
            n_heads=num_heads,
            head_dim=dim // num_heads,
            min_freq=1.0,
            max_freq=100.0,
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads) for _ in range(depth)]
        )
        self.final_layer = nn.Sequential(
            nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(dim, 1, bias=False),
        )

    def forward(
        self,
        inputs,
        y_indices,
        y_offsets,
        timesteps,
        condition_repeats=None,
    ):
        batch_size, _, height, width = inputs.shape
        tokens = self.patch_embed(inputs.flatten(2).transpose(1, 2))
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        condition = (
            self.t_embedder(timesteps).unsqueeze(1)
            + self.y_embedder(y_indices, y_offsets).unsqueeze(1)
        )
        if condition_repeats is not None:
            condition = torch.cat(
                [condition[:count] for count in condition_repeats],
                dim=0,
            )
            if len(condition) != batch_size:
                raise ValueError(
                    "condition_repeats must sum to the discriminator batch size"
                )

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                tokens, _ = checkpoint(
                    block,
                    tokens,
                    condition,
                    height,
                    width,
                    self.rope,
                    1,
                    None,
                    None,
                    use_reentrant=False,
                )
            else:
                tokens, _ = block(
                    tokens,
                    condition,
                    height,
                    width,
                    self.rope,
                    num_cls_tokens=1,
                )

        return self.final_layer(tokens[:, 0]).flatten()


def discriminator_losses(
    logits_real,
    logits_fake,
    logits_real_gp,
    logits_fake_gp,
    weighting,
    gp_scale,
    gp_eps,
    center_scale,
):
    logits_real = logits_real.float()
    logits_fake = logits_fake.float()
    logits_real_gp = logits_real_gp.float()
    logits_fake_gp = logits_fake_gp.float()
    gp_weight = gp_scale * weighting[:len(logits_real_gp)] / (gp_eps ** 2)
    loss_adv = F.softplus(-(logits_real - logits_fake)).mean()
    loss_r1 = (
        (logits_real_gp - logits_real[:len(logits_real_gp)]).square()
        * gp_weight
    ).mean()
    loss_r2 = (
        (logits_fake_gp - logits_fake[:len(logits_fake_gp)]).square()
        * gp_weight
    ).mean()
    loss_center = ((logits_real + logits_fake).square() * center_scale).mean()
    return {
        "total": loss_adv + loss_r1 + loss_r2 + loss_center,
        "adv": loss_adv,
        "r1": loss_r1,
        "r2": loss_r2,
        "center": loss_center,
    }


def generator_losses(
    logits_real,
    logits_fake,
    predicted_target,
    source,
    weighting,
    ot_scale,
):
    logits_real = logits_real.float()
    logits_fake = logits_fake.float()
    loss_adv = F.softplus(-(logits_fake - logits_real)).mean()
    loss_ot = (
        (predicted_target.float() - source.float())
        .square()
        .mean(dim=(1, 2, 3))
        .mul(ot_scale / weighting)
        .mean()
    )
    return {
        "total": loss_adv + loss_ot,
        "adv": loss_adv,
        "ot": loss_ot,
    }


@torch.no_grad()
def sample_adversarial_flow(
    model,
    tag_processor,
    latent_size,
    batch_size,
    prompts,
    device,
    steps=1,
    guidance_scale=1.0,
    noise=None,
):
    """Sample a model trained on designated AFM transitions."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")

    in_channels = model.in_channels
    if isinstance(latent_size, (tuple, list)):
        height, width = latent_size
    else:
        height = width = latent_size

    if noise is None:
        samples = torch.randn(
            batch_size,
            in_channels,
            height,
            width,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
    else:
        samples = noise[:batch_size].clone().to(
            device=device,
            dtype=next(model.parameters()).dtype,
        )

    y_indices, y_offsets = tag_processor.process_prompts(
        prompts[:batch_size],
        device,
    )
    use_guidance = guidance_scale != 1.0
    if use_guidance:
        null_indices, null_offsets = tag_processor.process_prompts(
            [""] * batch_size,
            device,
        )

    was_training = model.training
    model.eval()
    for index in range(steps):
        timestep = 1.0 - index / steps
        timesteps = torch.full(
            (batch_size,),
            timestep,
            device=device,
            dtype=torch.float32,
        )
        predicted = model(samples, timesteps, y_indices, y_offsets)
        if use_guidance:
            predicted_null = model(
                samples,
                timesteps,
                null_indices,
                null_offsets,
            )
            predicted = predicted_null + guidance_scale * (
                predicted - predicted_null
            )
        samples = predicted

    model.train(was_training)
    return samples
