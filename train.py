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
import torch.nn.functional as F
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

    # Save EMA state if available
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.shadow

    # Atomic write
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

    # Filter by shape
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

    # Restore optimizer
    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(">>> Optimizer state restored.")
        except Exception as e:
            print(f">>> Could not restore optimizer state: {e}")

    # Restore EMA
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
    """Exponential Moving Average for model parameters."""
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
def setup_vae(config: Dict[str, Any], device):
    use_vae = config['model'].get('use_vae', False)
    use_tiny_vae = config['model'].get('use_tiny_vae', False)
    if not (use_vae or use_tiny_vae):
        return None, None, None, 3

    stats_path = "vae_stats.pt"
    standard_vae = None

    if os.path.exists(stats_path):
        print(f">>> Loading VAE normalization stats from {stats_path}...")
        stats = torch.load(stats_path, map_location='cpu', weights_only=False)
        latents_mean = stats["mean"].to(device=device, dtype=torch.bfloat16)
        latents_std = stats["std"].to(device=device, dtype=torch.bfloat16)
    else:
        from diffusers import AutoencoderKLFlux2
        print(">>> Loading Standard FLUX.2 VAE to extract normalization stats...")
        standard_vae = AutoencoderKLFlux2.from_pretrained(
            "black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.bfloat16
        ).to(device=device).eval()
        latents_mean = standard_vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype=torch.bfloat16)
        latents_std = torch.sqrt(
            standard_vae.bn.running_var.view(1, -1, 1, 1) + standard_vae.config.batch_norm_eps
        ).to(device, dtype=torch.bfloat16)
        torch.save({"mean": latents_mean.cpu(), "std": latents_std.cpu()}, stats_path)
        print(f">>> Saved VAE normalization stats to {stats_path}")

    if use_tiny_vae:
        if standard_vae is not None:
            del standard_vae
            torch.cuda.empty_cache()
        from flux2_tiny_autoencoder import Flux2TinyAutoEncoder
        print(">>> Loading Tiny FLUX.2 VAE...")
        vae = Flux2TinyAutoEncoder.from_pretrained("fal/FLUX.2-Tiny-AutoEncoder")
        vae = vae.to(device=device, dtype=torch.bfloat16).eval()
        in_channels = 128
        print(f">>> Tiny VAE Mode: in_channels = {in_channels}")
    else:
        if standard_vae is None:
            from diffusers import AutoencoderKLFlux2
            print(">>> Loading Standard FLUX.2 VAE...")
            standard_vae = AutoencoderKLFlux2.from_pretrained(
                "black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.bfloat16
            ).to(device=device).eval()
        vae = standard_vae
        in_channels = 128
        print(f">>> Standard VAE Mode: in_channels = {in_channels}")

    return vae, latents_mean, latents_std, in_channels


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
    return torch.arccos(torch.cos(torch.tensor(0.0)) * (1 - u) + torch.cos(torch.tensor(math.pi / 2)) * u) / (math.pi / 2)


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
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    # Apply to optimizer
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

    from model_dit import TokenformerDiT, TagProcessor, sample_flow
    ModelClass, TagProcessor, sample_fn = TokenformerDiT, TagProcessor, sample_flow

    tag_processor = TagProcessor("tags.txt")
    num_classes = tag_processor.num_classes

    # DataLoader
    use_cached_latents = config['data'].get('use_cached_latents', False)
    wds_loader = WDSLoader(
        url=config['data'].get('cache_dir' if use_cached_latents else 'webdataset_url'),
        csv_path=config['data'].get('csv_path'),
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        use_advanced_captions=config['data'].get('use_advanced_captions', True)
    )
    dataloader = wds_loader.make_loader()
    data_iter = iter(dataloader)

    image_size = config['training']['image_size']

    # VAE
    vae, latents_mean, latents_std, in_channels = setup_vae(config, device)
    use_vae = vae is not None
    latent_size = (image_size // 8) if use_vae else image_size // config['model'].get('patch_size', 16)

    # Model
    model = ModelClass(
        in_channels=in_channels if use_vae else config['model'].get('in_channels', 3),
        dim=config['model'].get('dim', 768),
        depth=config['model'].get('depth', 12),
        num_heads=config['model'].get('num_heads', 12),
        num_classes=num_classes,
        use_checkpoint=config['training'].get('gradient_checkpointing', False),
        fcdm_blocks=config['model'].get('fcdm_blocks', 2),
    ).to(device)

    # torch.compile BEFORE DDP
    if config['training'].get('compile', False):
        print(">>> Compiling Model...")
        model = torch.compile(model, mode="max-autotune")

    # Resume
    global_step = 0
    fixed_prompts = None
    fixed_noise = None
    resume_dir = config.get('resume_from', "outputs_dit/")

    # EMA (before DDP)
    use_ema = config['training'].get('use_ema', True)
    ema = EMA(model, decay=config['training'].get('ema_decay', 0.9999)) if use_ema else None

    if resume_dir and os.path.isdir(resume_dir):
        ckpt_files = glob.glob(os.path.join(resume_dir, "ckpt_step_*.pth"))
        if ckpt_files:
            resume_path = sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            global_step, fixed_prompts, fixed_noise = load_checkpoint(
                model, None, ema, resume_path, device, is_ddp=False
            )

    # DDP wrap AFTER compile
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    model_raw = model.module if hasattr(model, 'module') else model

    # Optimizer
    optimizer = create_optimizer(model, config)

    # Restore optimizer after construction
    if resume_dir and os.path.isdir(resume_dir):
        ckpt_files = glob.glob(os.path.join(resume_dir, "ckpt_step_*.pth"))
        if ckpt_files:
            resume_path = sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as e:
                    print(f">>> Could not restore optimizer state: {e}")

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
    if use_vae and config['training'].get('compile_vae', False):
        print(">>> Compiling VAE...")
        vae = torch.compile(vae)

    # WandB
    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-C2I'), config=config)
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

            # Encode
            with torch.no_grad():
                if use_cached_latents:
                    latents = images.to(device=device, dtype=torch.bfloat16)
                    vae_channels = latents.shape[1]
                    if latents.shape[1] != in_channels:
                        latents = F.pixel_unshuffle(latents, 2)
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
                    vae_channels = latents.shape[1]
                    if latents.shape[1] != in_channels:
                        latents = F.pixel_unshuffle(latents, 2)
                    latents = (latents - latents_mean) / latents_std
                    inputs = latents.to(dtype=torch.bfloat16)
                else:
                    inputs = images.to(device, memory_format=torch.channels_last).to(dtype=torch.bfloat16)

            # Fixed validation data (deterministic sizing)
            if rank == 0 and fixed_prompts is None:
                n_fixed = min(16, len(prompts))
                fixed_prompts = prompts[:n_fixed]
                latent_size = inputs.shape[-1]
                fixed_noise = torch.randn(n_fixed, in_channels, latent_size, latent_size,
                                          device=device, dtype=torch.bfloat16)

            B, C, H, W = inputs.shape

            # Sample timesteps
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

                log_dict = {
                    "train/fm_loss": avg_metrics['fm_loss'],
                    "train/neg_loss": avg_metrics['neg_loss'],
                    "train/deltafm_loss": avg_metrics['deltafm_loss'],
                    "train/total_loss": avg_metrics['total_loss'],
                    "train/infonce_loss": avg_metrics['infonce_loss'],
                    "train/deltafm_lambda": deltafm_lambda,
                    "train/lr": optimizer.param_groups[0]['lr'],
                }
                wandb.log(log_dict, step=global_step)

                pbar.set_postfix({
                    "fm": f"{avg_metrics['fm_loss']:.4f}",
                    "dfm": f"{avg_metrics['deltafm_loss']:.4f}",
                    "total": f"{avg_metrics['total_loss']:.4f}",
                    "infonce": f"{avg_metrics['infonce_loss']:.4f}",
                })

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
                    samples = sample_fn(
                        model_raw,
                        tag_processor,
                        latent_size,
                        len(fixed_prompts),
                        fixed_prompts,
                        device,
                        guidance_scale=config['training'].get('cfg_scale', 1.4),
                        noise=fixed_noise,
                    )

                    if use_vae:
                        samples = samples.to(dtype=torch.bfloat16)
                        if 'vae_channels' in locals() and samples.shape[1] != vae_channels:
                            samples = F.pixel_shuffle(samples, 2)
                        recon = vae.decode(samples).sample if hasattr(vae, 'decode') else vae(samples)
                        samples = recon.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)

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
