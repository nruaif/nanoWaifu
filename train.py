import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import argparse
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import wandb
import glob
import builtins
from dataset import WDSLoader
from emf import emf_loss, sample_emf, sample_emf_times
import torch.nn.functional as F
from flux2_tiny_autoencoder import normalize_latent_f8
from huggingface_hub import upload_file
import threading
import time
from contextlib import nullcontext
from typing import Optional, Dict, Any
import math


# =============================================================================
# HF Hub Async Upload
# =============================================================================
def _async_upload(ckpt_path: str, repo_id: str, step: int):
    try:
        upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=os.path.basename(ckpt_path),
            repo_id=repo_id,
            commit_message=f"Checkpoint at step {step}"
        )
        print(f"[HF] Uploaded: {ckpt_path}")
    except Exception as e:
        print(f"[HF] Upload failed for {ckpt_path}: {e}")


# =============================================================================
# Distributed Setup
# =============================================================================
def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, local_rank, world_size, device
    else:
        return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Checkpointing (with EMA support)
# =============================================================================
def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    ckpt_files = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    if not ckpt_files:
        return None
    return sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]


def cleanup_checkpoints(output_dir: str, max_checkpoints: int, rank: int):
    if rank != 0 or max_checkpoints <= 0:
        return
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoints) > max_checkpoints:
        for ckpt in checkpoints[:-max_checkpoints]:
            try:
                os.remove(ckpt)
                print(f"Removed old checkpoint: {ckpt}")
            except OSError as e:
                print(f"Error removing {ckpt}: {e}")


def save_checkpoint(
    model,
    optimizer,
    ema,
    rank: int,
    output_dir: str,
    step: int,
    config: Dict[str, Any],
    fixed_prompts=None,
    fixed_noise=None,
    push_to_hf: bool = True,
    repo_id: Optional[str] = None,
    is_ddp: bool = False,
):
    if rank != 0:
        return

    print(f"\n[Step {step}] Saving Checkpoint...")

    model_to_save = model.module if is_ddp else model

    ckpt_path = os.path.join(output_dir, f'ckpt_step_{step}.pth')

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step,
        "config": config,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "fixed_prompts": fixed_prompts,
        "fixed_noise": fixed_noise,
        "timestamp": time.time(),
    }

    if ema is not None:
        checkpoint["ema_state_dict"] = ema.shadow

    temp_path = ckpt_path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, ckpt_path)

    cleanup_checkpoints(output_dir, config.get('max_checkpoints', 3), rank)
    print(f"Checkpoint saved: {ckpt_path}")

    if push_to_hf and repo_id is not None:
        thread = threading.Thread(
            target=_async_upload,
            args=(ckpt_path, repo_id, step),
            daemon=True
        )
        thread.start()


def load_checkpoint(
    model,
    optimizer,
    ema,
    resume_path: str,
    device,
    is_ddp: bool = False,
) -> tuple[int, Optional[Any], Optional[Any]]:
    """Load checkpoint with EMA and shape mismatch handling."""
    if not resume_path or not os.path.exists(resume_path):
        return 0, None, None

    print(f"Resuming from: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

    model_to_load = model.module if is_ddp else model

    state_dict = checkpoint["model_state_dict"]
    model_state = model_to_load.state_dict()

    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                print(f">>> Shape Mismatch: Skipping {k}. "
                      f"Checkpoint: {v.shape}, Model: {model_state[k].shape}")
        else:
            print(f">>> Key not in model: {k}")

    missing_keys = [k for k in model_state if k not in filtered_state]
    if missing_keys:
        print(f">>> Missing keys (will init): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")

    model_to_load.load_state_dict(filtered_state, strict=False)

    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(">>> Optimizer state restored.")
        except Exception as e:
            print(f">>> Could not restore optimizer state: {e}")

    if ema is not None and "ema_state_dict" in checkpoint:
        try:
            ema.shadow = {k: v.to(device) for k, v in checkpoint["ema_state_dict"].items()}
            print(">>> EMA state restored.")
        except Exception as e:
            print(f">>> Could not restore EMA state: {e}")

    global_step = checkpoint.get("global_step", 0)
    fixed_noise = checkpoint.get("fixed_noise")
    fixed_prompts = checkpoint.get("fixed_prompts")
    if fixed_noise is not None:
        fixed_noise = fixed_noise.to(device)

    return global_step, fixed_prompts, fixed_noise


# =============================================================================
# Optimizer
# =============================================================================
def create_optimizer(model, config: Dict[str, Any]):
    lr = config['training']['learning_rate']
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))


# =============================================================================
# EMA
# =============================================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# VAE Setup
# =============================================================================
def load_vae_stats(device):
    """Load (or extract once and cache) FLUX.2 latent normalization stats.

    Stats are stored in the VAE's internal 2x2-patchified layout [1, 4C, 1, 1]
    (e.g. 128 channels for the 32-channel f/8 latent).
    """
    stats_path = "vae_stats.pt"
    if os.path.exists(stats_path):
        print(f">>> Loading VAE normalization stats from {stats_path}...")
        stats = torch.load(stats_path, map_location='cpu', weights_only=False)
        return (stats["mean"].to(device=device, dtype=torch.bfloat16),
                stats["std"].to(device=device, dtype=torch.bfloat16))

    from diffusers import AutoencoderKLFlux2
    print(">>> Loading Standard FLUX.2 VAE to extract normalization stats...")
    std_vae = AutoencoderKLFlux2.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device=device).eval()
    mean = std_vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype=torch.bfloat16)
    std = torch.sqrt(
        std_vae.bn.running_var.view(1, -1, 1, 1) + std_vae.config.batch_norm_eps
    ).to(device, dtype=torch.bfloat16)
    torch.save({"mean": mean.cpu(), "std": std.cpu()}, stats_path)
    print(f">>> Saved VAE normalization stats to {stats_path}")
    del std_vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mean, std


def load_decode_vae(config: Dict[str, Any], device):
    """Load the VAE used for decoding validation samples.

    Returns (vae, decode_channels): the Tiny VAE decodes the patchified
    128-channel f/16 layout, the standard VAE decodes 32-channel f/8 directly.
    """
    if config['model'].get('use_tiny_vae', False):
        from flux2_tiny_autoencoder import Flux2TinyAutoEncoder
        vae = Flux2TinyAutoEncoder.from_pretrained("fal/FLUX.2-Tiny-AutoEncoder")
        vae = vae.to(device=device, dtype=torch.bfloat16).eval()
        return vae, vae.tiny_vae.latent_channels * 4
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device=device).eval()
    return vae, vae.config.latent_channels


def setup_vae(config: Dict[str, Any], device, decode_only: bool = False):
    """Returns (vae, latents_mean, latents_std, in_channels, decode_channels).

    The model always consumes the native f/8 latent (in_channels = 32 for
    FLUX.2). With decode_only (cached-latent training) no VAE is kept in
    memory during training; load_decode_vae() provides one at validation.
    """
    use_vae = config['model'].get('use_vae', False)
    use_tiny_vae = config['model'].get('use_tiny_vae', False)
    if not (use_vae or use_tiny_vae):
        return None, None, None, 3, None

    latents_mean, latents_std = load_vae_stats(device)
    in_channels = latents_mean.shape[1] // 4  # f/8 native channels
    print(f">>> VAE Mode ({'tiny' if use_tiny_vae else 'standard'}): "
          f"model consumes f/8 latents with {in_channels} channels")

    if decode_only:
        print(">>> Cached-latent training: VAE loads lazily at validation")
        return None, latents_mean, latents_std, in_channels, None

    vae, decode_channels = load_decode_vae(config, device)
    return vae, latents_mean, latents_std, in_channels, decode_channels


# =============================================================================
# Timestep Samplers
# =============================================================================
def sample_logit_normal(m_loc: float, s_scale: float, bs: int, device, dtype):
    eps = torch.randn(bs, device=device, dtype=dtype)
    return torch.sigmoid(m_loc + s_scale * eps)


def sample_uniform(bs: int, device, dtype):
    return torch.rand(bs, device=device, dtype=dtype)


def sample_cosine(bs: int, device, dtype):
    u = torch.rand(bs, device=device, dtype=dtype)
    return 2.0 * torch.arccos(1 - u) / math.pi


# =============================================================================
# Loss Functions
# =============================================================================
def compute_fm_loss(x0_pred, x0_target, t, loss_type="mse"):
    B = x0_pred.shape[0]
    t_reshaped = t.view(B, 1, 1, 1)
    if loss_type == "mse":
        loss = F.mse_loss(x0_pred.float(), x0_target.float(), reduction='none')
    elif loss_type == "l1":
        loss = F.l1_loss(x0_pred.float(), x0_target.float(), reduction='none')
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(x0_pred.float(), x0_target.float(), reduction='none', beta=0.1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    snr = (1 - t_reshaped)**2 / (t_reshaped**2 + 1e-5)
    return (loss * snr).mean()


def compute_deltafm_loss(x0_pred, x0_target, t, deltafm_lambda: float):
    B = x0_pred.shape[0]
    if B <= 1 or deltafm_lambda <= 0:
        fm_loss = compute_fm_loss(x0_pred, x0_target, t)
        return fm_loss, torch.zeros((), device=x0_pred.device), fm_loss
    perm = torch.arange(B, device=x0_pred.device).roll(1)
    inputs_neg = x0_target[perm]
    fm_loss = compute_fm_loss(x0_pred, x0_target, t)
    neg_loss = compute_fm_loss(x0_pred, inputs_neg, t)
    deltafm_loss = fm_loss - deltafm_lambda * neg_loss
    return fm_loss, neg_loss, deltafm_loss


# =============================================================================
# Augmentations
# =============================================================================
def apply_noise_injection(xt, noise_inject_ratio: float):
    if noise_inject_ratio <= 0:
        return xt
    noise_mask = (torch.rand(xt.shape[0], 1, 1, 1, device=xt.device) < 0.5).to(xt.dtype)
    noise_injection = torch.randn_like(xt)
    return xt + noise_mask * noise_inject_ratio * noise_injection


def apply_cross_sample_mixing(xt, inputs, t, cross_ratio: float):
    if cross_ratio <= 0:
        return xt
    B = xt.shape[0]
    cross_mask = (torch.rand(B, 1, 1, 1, device=xt.device) < 0.5).to(xt.dtype)
    inputs_neg = inputs.roll(shifts=1, dims=0)
    noise_neg = torch.randn_like(inputs)
    t_reshaped = t.view(B, 1, 1, 1)
    xt_neg = (1 - t_reshaped) * inputs_neg + t_reshaped * noise_neg
    return xt + cross_mask * cross_ratio * (xt_neg - xt)


# =============================================================================
# LR Scheduler
# =============================================================================
def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)]
    return schedulers


# =============================================================================
# Main Training
# =============================================================================
def train(config_path: str):
    is_ddp, rank, local_rank, world_size, device = setup_distributed()

    torch.autograd.set_detect_anomaly(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.enable_flash_sdp(True)

    if rank != 0:
        builtins.print = lambda *args, **kwargs: None

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    from model_dit import FCDM, TagProcessor, sample_flow

    tag_processor = TagProcessor("tags.txt")
    num_classes = tag_processor.num_classes

    # DataLoader
    use_cached_latents = config['data'].get('use_cached_latents', False)
    wds_loader = WDSLoader(
        url=config['data'].get('cache_dir' if use_cached_latents else 'webdataset_url'),
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        use_advanced_captions=config['data'].get('use_advanced_captions', True),
        fast_loading=config['data'].get('fast_loading', False)
    )
    dataloader = wds_loader.make_loader()
    data_iter = iter(dataloader)

    image_size = config['training']['image_size']

    # VAE
    vae, latents_mean, latents_std, in_channels, vae_channels = setup_vae(
        config, device, decode_only=use_cached_latents
    )
    use_vae = vae is not None or latents_mean is not None
    latent_size = (image_size // 8) if use_vae else image_size // config['model'].get('patch_size', 16)

    # Model
    model = FCDM(
        in_channels=in_channels if use_vae else config['model'].get('in_channels', 3),
        dim=config['model'].get('dim', 128),
        depth=config['model'].get('depth', 2),
        num_classes=num_classes,
        use_checkpoint=config['training'].get('gradient_checkpointing', False),
        attn_every=config['model'].get('attn_every', 3),
        use_lfq_memory=config['model'].get('use_lfq_memory', False),
        lfq_c_mem=config['model'].get('lfq_c_mem', 512),
        lfq_num_latents=config['model'].get('lfq_num_latents', 8),
        lfq_k_bits=config['model'].get('lfq_k_bits', 12),
        use_topk_memory=config['model'].get('use_topk_memory', False),
        topk_c_mem=config['model'].get('topk_c_mem', 512),
        topk_num_entries=config['model'].get('topk_num_entries', 4096),
        topk_top_k=config['model'].get('topk_top_k', 8),
        topk_query_dim=config['model'].get('topk_query_dim', 64),
    ).to(device)

    # torch.compile BEFORE DDP
    if config['training'].get('compile', False):
        print(">>> Compiling Model...")
        model = torch.compile(model, mode="max-autotune")

    # Resume
    resume_dir = config.get('resume_from', "outputs_fcdm/")

    # EMA (before DDP)
    use_ema = config['training'].get('use_ema', True)
    ema = EMA(model, decay=config['training'].get('ema_decay', 0.9999)) if use_ema else None

    # DDP wrap AFTER compile
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    model_raw = model.module if hasattr(model, 'module') else model

    # Optimizer
    optimizer = create_optimizer(model, config)

    # Restore model, EMA, and optimizer state from the newest checkpoint
    global_step = 0
    fixed_prompts = None
    fixed_noise = None
    resume_path = find_latest_checkpoint(resume_dir) if resume_dir else None
    if resume_path:
        global_step, fixed_prompts, fixed_noise = load_checkpoint(
            model, optimizer, ema, resume_path, device, is_ddp=is_ddp
        )

    # LR Scheduler
    schedulers = None
    if config['training'].get('use_lr_scheduler', False):
        schedulers = get_lr_scheduler(
            optimizer,
            warmup_steps=config['training'].get('warmup_steps', 10000),
            max_steps=config['training'].get('max_train_steps', 1000000),
            min_lr_ratio=config['training'].get('min_lr_ratio', 0.1)
        )

    # Compile VAE
    if vae is not None and config['training'].get('compile_vae', False):
        print(">>> Compiling VAE...")
        vae = torch.compile(vae)

    # WandB
    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'fcdm-training'), config=config)
        pbar = tqdm(range(global_step, config['training'].get('max_train_steps', 1000000)),
                    desc="Training", dynamic_ncols=True)
        os.makedirs(config['training']['output_dir'], exist_ok=True)

    # Training state
    accum_steps = max(1, config['training'].get('grad_accum_steps', 1))
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)
    log_every = config['training']['log_every_steps']
    save_every = config['training']['save_image_every_steps']

    deltafm_lambda = config['training'].get('deltafm_lambda', 0.05)
    infonce_weight = config['training'].get('infonce_weight', 0.2)
    noise_inject_ratio = config['training'].get('noise_inject_ratio', 0.1)
    cross_ratio = config['training'].get('cross_ratio', 0.1)
    loss_type = config['training'].get('loss_type', 'mse')
    timestep_sampler = config['training'].get('timestep_sampler', 'uniform')

    # Objective: "flow_matching" (multi-step) or "emf" (Euler Mean Flows,
    # one-step / few-step generation; see emf.py)
    objective = config['training'].get('objective', 'flow_matching')
    emf_opts = config['training'].get('emf', {}) or {}
    emf_delta_t = emf_opts.get('delta_t', 0.05)
    emf_interval_ratio = emf_opts.get('interval_ratio', 0.25)
    emf_time_sampler = emf_opts.get('time_sampler', 'uniform')
    emf_cfg_scale = emf_opts.get('cfg_scale', 2.5)
    emf_cfg_k = emf_opts.get('cfg_k', 0.4)
    emf_adaptive_c = emf_opts.get('adaptive_c', 1e-3)
    emf_adaptive_p = emf_opts.get('adaptive_p', 1.0)
    emf_val_steps = emf_opts.get('val_steps', 1)

    if objective == 'emf':
        running_metrics = {'emf_loss': 0.0}
    else:
        running_metrics = {
            'fm_loss': 0.0, 'neg_loss': 0.0, 'deltafm_loss': 0.0,
            'total_loss': 0.0, 'infonce_loss': 0.0,
        }

    while global_step < config['training'].get('max_train_steps', 1000000):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        step_metrics = {k: 0.0 for k in running_metrics}

        for accum_idx in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            images, prompts, _ = batch
            y_indices, y_offsets = tag_processor.process_prompts(
                prompts, device, dropout_prob=config['training'].get('class_dropout_prob', 0.1)
            )
            if objective == 'emf':
                y_null = tag_processor.process_prompts([""] * len(prompts), device)

            # Encode
            with torch.no_grad():
                if use_cached_latents:
                    latents = images.to(device=device, dtype=torch.bfloat16)
                    if latents.shape[1] != in_channels:
                        raise ValueError(
                            f"Cached latents have {latents.shape[1]} channels but this config "
                            f"expects f/8 latents with {in_channels} channels. Old caches use "
                            f"the f/16 patchified format - re-run cache_latents.py."
                        )
                    inputs = latents
                elif use_vae:
                    images = images.to(device, memory_format=torch.channels_last)
                    v_images = images.to(dtype=torch.bfloat16)
                    if hasattr(vae, 'encode'):
                        out = vae.encode(v_images)
                        if hasattr(out, 'latent_dist'):
                            latents = out.latent_dist.mode()
                        else:
                            latents = out[0] if isinstance(out, tuple) else out
                    else:
                        latents = vae(v_images)
                    if latents.shape[1] == latents_mean.shape[1]:
                        # Patchified f/16 output (Tiny VAE): normalize in the
                        # stats layout, then convert to the f/8 native layout
                        latents = (latents - latents_mean) / latents_std
                        latents = F.pixel_shuffle(latents, 2)
                    else:
                        # Native f/8 output (standard VAE)
                        latents = normalize_latent_f8(latents, latents_mean, latents_std)
                    inputs = latents.to(dtype=torch.bfloat16)
                else:
                    inputs = images.to(device, memory_format=torch.channels_last).to(dtype=torch.bfloat16)

            # Fixed validation data
            if rank == 0 and fixed_prompts is None:
                n_fixed = min(16, len(prompts))
                fixed_prompts = prompts[:n_fixed]
                latent_size = (inputs.shape[-2], inputs.shape[-1])
                fixed_noise = torch.randn(n_fixed, in_channels, latent_size[0], latent_size[1],
                                          device=device, dtype=torch.bfloat16)

            B, C, H, W = inputs.shape

            if objective == 'emf':
                # Euler Mean Flow (paper convention: t=0 noise, t=1 data)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext():
                    t, r = sample_emf_times(
                        B, device, dist=emf_time_sampler, interval_ratio=emf_interval_ratio
                    )
                    noise = torch.randn_like(inputs)
                    xt = (1 - t.view(B, 1, 1, 1)) * noise + t.view(B, 1, 1, 1) * inputs
                    total_loss, emf_metrics = emf_loss(
                        model, xt, inputs, t, r,
                        (y_indices, y_offsets), y_null,
                        delta_t=emf_delta_t, cfg_scale=emf_cfg_scale, cfg_k=emf_cfg_k,
                        adaptive_c=emf_adaptive_c, adaptive_p=emf_adaptive_p,
                    )

                loss = total_loss / accum_steps
                loss.backward()
                step_metrics['emf_loss'] += emf_metrics['loss'] / accum_steps
            else:
                # Sample timesteps (t=1 noise, t=0 data)
                if timestep_sampler == 'logit_normal':
                    t = sample_logit_normal(0.8, 1.0, B, device, torch.bfloat16)
                elif timestep_sampler == 'cosine':
                    t = sample_cosine(B, device, torch.bfloat16)
                else:
                    t = sample_uniform(B, device, torch.bfloat16)

                t_reshaped = t.view(B, 1, 1, 1)
                noise = torch.randn_like(inputs)
                xt = (1 - t_reshaped) * inputs + t_reshaped * noise

                # Augmentations
                xt = apply_noise_injection(xt, noise_inject_ratio)
                xt = apply_cross_sample_mixing(xt, inputs, t, cross_ratio)

                # Forward with autocast
                with torch.cuda.amp.autocast(dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext():
                    x0_pred, infonce_loss = model(xt, t, y_indices, y_offsets, return_layer_match=True)

                # Loss
                fm_loss, neg_loss, deltafm_loss = compute_deltafm_loss(
                    x0_pred, inputs, t, deltafm_lambda
                )
                total_loss = deltafm_loss + infonce_weight * infonce_loss

                loss = total_loss / accum_steps
                loss.backward()

                step_metrics['fm_loss'] += fm_loss.detach().item() / accum_steps
                step_metrics['neg_loss'] += neg_loss.detach().item() / accum_steps
                step_metrics['deltafm_loss'] += deltafm_loss.detach().item() / accum_steps
                step_metrics['total_loss'] += total_loss.detach().item() / accum_steps
                step_metrics['infonce_loss'] += infonce_loss.detach().item() / accum_steps

        # Step
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if ema is not None:
            ema.update()

        if schedulers is not None:
            for s in schedulers:
                s.step()

        global_step += 1

        for k in running_metrics:
            running_metrics[k] += step_metrics[k]

        # Logging
        if rank == 0:
            pbar.update(1)

            if global_step % log_every == 0:
                log_interval = log_every
                avg_metrics = {k: v / log_interval for k, v in running_metrics.items()}

                if objective == 'emf':
                    log_dict = {
                        "train/emf_loss": avg_metrics['emf_loss'],
                        "train/lr": optimizer.param_groups[0]['lr'],
                    }
                    pbar.set_postfix({"emf": f"{avg_metrics['emf_loss']:.4f}"})
                else:
                    log_dict = {
                        "train/fm_loss": avg_metrics['fm_loss'],
                        "train/neg_loss": avg_metrics['neg_loss'],
                        "train/deltafm_loss": avg_metrics['deltafm_loss'],
                        "train/total_loss": avg_metrics['total_loss'],
                        "train/infonce_loss": avg_metrics['infonce_loss'],
                        "train/deltafm_lambda": deltafm_lambda,
                        "train/lr": optimizer.param_groups[0]['lr'],
                    }
                    pbar.set_postfix({
                        "fm": f"{avg_metrics['fm_loss']:.4f}",
                        "dfm": f"{avg_metrics['deltafm_loss']:.4f}",
                        "total": f"{avg_metrics['total_loss']:.4f}",
                        "infonce": f"{avg_metrics['infonce_loss']:.4f}",
                    })
                wandb.log(log_dict, step=global_step)

                for k in running_metrics:
                    running_metrics[k] = 0.0

            # Validation & checkpoint
            if global_step % save_every == 0:
                save_checkpoint(
                    model, optimizer, ema, rank, config['training']['output_dir'],
                    global_step, config, fixed_prompts, fixed_noise,
                    push_to_hf=config.get('push_to_hf', True),
                    repo_id=config.get('hf_repo_id', None),
                    is_ddp=is_ddp,
                )

                print(f"\n[Step {global_step}] Generating validation samples...")
                model.eval()

                if ema is not None:
                    ema.apply_shadow()

                with torch.no_grad():
                    if objective == 'emf':
                        samples = sample_emf(
                            model_raw, tag_processor, latent_size, len(fixed_prompts),
                            fixed_prompts, device, steps=emf_val_steps, noise=fixed_noise,
                        )
                    else:
                        samples = sample_flow(
                            model_raw, tag_processor, latent_size, len(fixed_prompts),
                            fixed_prompts, device,
                            guidance_scale=config['training'].get('cfg_scale', 1.4),
                            noise=fixed_noise,
                        )

                    if use_vae:
                        lazy_vae = vae is None
                        decode_vae = vae
                        if decode_vae is None:
                            # Cached-latent training: VAE is only needed here
                            decode_vae, vae_channels = load_decode_vae(config, device)
                        samples = samples.to(dtype=torch.bfloat16)
                        if samples.shape[1] != vae_channels:
                            # f/8 (32ch) -> patchified f/16 (128ch) for Tiny VAE
                            samples = F.pixel_unshuffle(samples, 2)
                        recon = decode_vae.decode(samples).sample if hasattr(decode_vae, 'decode') else decode_vae(samples)
                        samples = recon.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)
                        if lazy_vae:
                            del decode_vae
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    grid = make_grid(samples, nrow=4)
                    save_path = os.path.join(config['training']['output_dir'], f"val_ema_step_{global_step}.png")
                    save_image(samples, save_path, nrow=4)
                    wandb.log({
                        "val/samples": wandb.Image(grid, caption=f"Step {global_step}")
                    }, step=global_step)

                if ema is not None:
                    ema.restore()
                model.train()

    # Final save
    if rank == 0:
        save_checkpoint(
            model, optimizer, ema, rank, config['training']['output_dir'],
            global_step, config, fixed_prompts, fixed_noise,
            push_to_hf=config.get('push_to_hf', True),
            repo_id=config.get('hf_repo_id', None),
            is_ddp=is_ddp,
        )
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)