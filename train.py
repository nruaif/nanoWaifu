import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import argparse
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
import glob
import builtins
from dataset import WDSLoader
import torch.nn.functional as F
import bitsandbytes as bnb
from huggingface_hub import upload_file
import threading


def _async_upload(ckpt_path, repo_id, step):
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


torch.backends.cuda.enable_flash_sdp(True)


def disp_loss(Z, tau):
    if Z.shape[0] <= 1:
        return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    Z_flat = Z.reshape(Z.shape[0], -1)
    D = torch.pdist(Z_flat, p=2) ** 2
    # Add epsilon to prevent log(0)
    return torch.log(torch.mean(torch.exp(-D / tau)) + 1e-8)


# Dynamic model import
def get_model_and_sampler(config):
    model_type = config['model'].get('type', 'v2')

    if True:
        from model_dit import TokenformerDiT as ModelClass, sample_flow as sample_fn
        # We'll use the TagProcessor from model_v2 as it's compatible
        from model_v2 import TagProcessor
        print(">>> Training TokenformerDiT Model")
    elif config['model'].get('use_v2_model', False) or model_type == 'v2':
        from model_v2 import FCDMV2 as ModelClass, TagProcessor, sample_flow as sample_fn
        print(">>> Training V2 Model (CSP + Hybrid ViT + ReLU^2 + Gated Skip)")
    else:
        from model import FCDM as ModelClass, TagProcessor, sample_flow as sample_fn
        print(">>> Training V1 Model (Baseline FCDM)")
    return ModelClass, TagProcessor, sample_fn


def setup_ddp():
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


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def cleanup_checkpoints(output_dir, max_checkpoints, rank):
    if rank != 0: return
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoints) > max_checkpoints:
        checkpoints_to_remove = checkpoints[:-max_checkpoints]
        for ckpt in checkpoints_to_remove:
            try:
                os.remove(ckpt)
                print(f"Removed old checkpoint: {ckpt}")
            except OSError as e:
                print(f"Error removing {ckpt}: {e}")


def save_checkpoint(
        model,
        optimizer,
        rank,
        output_dir,
        step,
        config,
        fixed_prompts=None,
        fixed_noise=None,
        push_to_hf=True,
        repo_id="Shio-Koube/ConvNext-Diff"
):
    if rank != 0:
        return

    print(f"\n[Step {step}] Saving Checkpoint...")

    model_to_save = model.module if hasattr(model, 'module') else model
    ckpt_path = os.path.join(output_dir, f'ckpt_step_{step}.pth')

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "global_step": step,
        "config": config,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "fixed_prompts": fixed_prompts,
        "fixed_noise": fixed_noise
    }

    torch.save(checkpoint, ckpt_path)

    # cleanup old checkpoints
    cleanup_checkpoints(output_dir, config.get('max_checkpoints', 3), rank)

    print(f"Checkpoint saved: {ckpt_path}")

    # Async upload
    if push_to_hf and repo_id is not None:
        thread = threading.Thread(
            target=_async_upload,
            args=(ckpt_path, repo_id, step),
            daemon=True
        )
        thread.start()


def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    # Performance Tuning
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    if rank != 0:
        def print_pass(*args, **kwargs): pass

        builtins.print = print_pass

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate model and helpers
    ModelClass, TagProcessor, sample_fn = get_model_and_sampler(config)

    tag_processor = TagProcessor("tags.txt")
    num_classes = tag_processor.num_classes

    # Load Data
    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'],
        csv_path=config['data'].get('csv_path'),
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        use_advanced_captions=config['data'].get('use_advanced_captions', True)
    )
    dataloader = wds_loader.make_loader()

    image_size = config['training']['image_size']

    use_vae = config['model'].get('use_vae', False)
    use_tiny_vae = config['model'].get('use_tiny_vae', False)
    in_channels = config['model'].get('in_channels', 3)

    # Load VAE and get Stats
    if use_vae or use_tiny_vae:
        from diffusers import AutoencoderKLFlux2
        print(">>> Loading Standard FLUX.2 VAE to extract normalization stats...")
        standard_vae = AutoencoderKLFlux2.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to(device=device).eval()

        # Pre-compute normalization stats from the Standard VAE's BatchNorm
        latents_mean = standard_vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        latents_std = torch.sqrt(standard_vae.bn.running_var.view(1, -1, 1, 1) + standard_vae.config.batch_norm_eps).to(
            device)

        if use_tiny_vae:
            # Free up standard VAE memory since we only needed the stats
            del standard_vae
            torch.cuda.empty_cache()

            from flux2_tiny_autoencoder import Flux2TinyAutoEncoder
            print(">>> Loading Tiny FLUX.2 VAE...")
            vae = Flux2TinyAutoEncoder.from_pretrained(
                "fal/FLUX.2-Tiny-AutoEncoder",
            ).to(device=device, dtype=torch.bfloat16).eval()

            in_channels = 128
            print(f">>> Tiny VAE Mode Enabled: Model in_channels adjusted to {in_channels}")

        else:
            vae = standard_vae
            # 32 native channels * 4 (from unshuffle factor 2) = 128 channels
            in_channels = 128
            print(f">>> VAE Mode Enabled: Model in_channels adjusted to {in_channels}")

    # Instantiate model
    if True:
        # Calculate latent size for RoPE
        # Both VAEs downsample spatial dimensions by 16 overall
        latent_size = (image_size // 16) if (use_vae or use_tiny_vae) else image_size // config['model'].get(
            'patch_size', 16)

        model = ModelClass(
            in_channels=in_channels,
            dim=config['model'].get('fcdm_dim', 768),
            depth=config['model'].get('fcdm_depth', 12),
            num_heads=config['model'].get('num_heads', 12),
            num_classes=num_classes,
            latent_size=latent_size,
            use_checkpoint=config['training'].get('gradient_checkpointing', False)
        ).to(device, memory_format=torch.channels_last)
    else:
        model = ModelClass(
            in_channels=in_channels,
            base_channels=config['model'].get('fcdm_dim', 128),
            num_blocks=config['model'].get('fcdm_depth', 2),
            num_classes=num_classes,
            patch_size=config['model'].get('patch_size', 16),
            use_t_cond=True,
            use_checkpoint=config['training'].get('gradient_checkpointing', False)
        ).to(device, memory_format=torch.channels_last)

    # Resume Logic
    global_step = 0
    resume_path = config.get('resume_from', "outputs_dit/")
    fixed_prompts = None
    fixed_noise = None

    if resume_path:
        if os.path.isdir(resume_path):
            ckpt_files = glob.glob(os.path.join(resume_path, "ckpt_step_*.pth"))
            if ckpt_files:
                resume_path = sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            else:
                resume_path = None

        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model_to_load = model.module if hasattr(model, 'module') else model

            # Filter mismatching shapes (e.g., when transitioning to VAE or changing dimensions)
            state_dict = checkpoint["model_state_dict"]
            model_state = model_to_load.state_dict()
            keys_to_delete = []
            for k in state_dict.keys():
                if k in model_state and state_dict[k].shape != model_state[k].shape:
                    print(
                        f">>> Shape Mismatch: Removing {k} from state_dict due to shape mismatch. Checkpoint: {state_dict[k].shape}, Model: {model_state[k].shape}")
                    keys_to_delete.append(k)

            for k in keys_to_delete:
                del state_dict[k]

            model_to_load.load_state_dict(state_dict, strict=False)
            global_step = checkpoint["global_step"] - 10
            if "fixed_noise" in checkpoint and checkpoint["fixed_noise"] is not None:
                fixed_noise = checkpoint["fixed_noise"].to(device)
            if "fixed_prompts" in checkpoint:
                fixed_prompts = checkpoint["fixed_prompts"]

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if config['training'].get('compile', False):
        print(">>> Compiling Model...")
        model = torch.compile(model)

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-C2I'), config=config)
        pbar = tqdm(range(global_step, config['training'].get('max_train_steps', 1000000)),
                    desc="Training", dynamic_ncols=True)
        os.makedirs(config['training']['output_dir'], exist_ok=True)

    data_iter = iter(dataloader)
    running_loss = 0.0
    accum_steps = config['training'].get('grad_accum_steps', 1)

    while global_step < config['training'].get('max_train_steps', 1000000):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            images, prompts, _ = batch
            images = images.to(device, memory_format=torch.channels_last)
            y_indices, y_offsets = tag_processor.process_prompts(
                prompts, device, dropout_prob=config['training'].get('class_dropout_prob', 0.1)
            )

            # VAE Encoding Logic
            if use_tiny_vae:
                with torch.no_grad():
                    v_images = images.to(dtype=torch.bfloat16)

                    # encode directly returns the needed latents structure
                    out = vae.encode(v_images, return_dict=False)
                    latents = out[0] if isinstance(out, tuple) else out

                    # Apply FLUX.2 specific normalization
                    latents = (latents - latents_mean) / latents_std

                    # No pixel_unshuffle needed for Tiny VAE
                    inputs = latents.to(dtype=torch.float32)

            elif use_vae:
                with torch.no_grad():
                    v_images = images.to(dtype=torch.bfloat16)

                    # Encode to 32-channel latent distribution
                    latents = vae.encode(v_images).latent_dist.mode()

                    # Apply FLUX.2 specific normalization
                    latents = (latents - latents_mean) / latents_std

                    # Unshuffle: 32 channels -> 128 channels (Height and Width are halved)
                    inputs = F.pixel_unshuffle(latents, 2).to(dtype=torch.float32)
            else:
                inputs = images

            if rank == 0 and fixed_prompts is None:
                # Capture initial validation samples
                fixed_prompts = prompts[:16]
                fixed_noise = torch.randn_like(inputs[:16])

            # --- Flow Matching / Rectified Flow Training ---
            B, C, H, W = inputs.shape

            # 1. Calculate effective dimension (m) and scaling factor (alpha)
            m = C * H * W
            n = 32768.0
            alpha = (m / n) ** 0.5

            # 2. Sample base uniform timestep
            t_base = torch.rand((B,), device=device)

            # 3. Apply Dimension-Dependent Shift formula
            t = (alpha * t_base) / (1.0 + (alpha - 1.0) * t_base)

            t_reshaped = t.view(-1, 1, 1, 1)
            noise = torch.randn_like(inputs)
            xt = (1 - t_reshaped) * inputs + t_reshaped * noise
            target = noise - inputs
            # -----------------------------------------------
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(xt, t, y_indices, y_offsets)
                # Scale loss by accumulation steps
                loss = F.mse_loss(pred, target)
                x0_pred = xt - t_reshaped * pred
                x0_neg = torch.roll(x0_pred, shifts=1, dims=0).detach()
                contrast_loss = F.mse_loss(x0_pred, x0_neg)
                loss = (loss) / accum_steps

            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        running_loss += loss_accum

        if rank == 0:
            pbar.update(1)
            if global_step % config['training']['log_every_steps'] == 0:
                avg_loss = running_loss / config['training']['log_every_steps']
                avg_contrast_loss = contrast_loss.item()
                wandb.log({"train/loss": avg_loss}, step=global_step)
                wandb.log({"train/contrast_loss": avg_contrast_loss}, step=global_step)

                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                running_loss = 0.0

            if global_step % config['training']['save_image_every_steps'] == 0:
                save_checkpoint(model, optimizer, rank, config['training']['output_dir'],
                                global_step, config, fixed_prompts, fixed_noise)

                print(f"\n[Step {global_step}] Generating validation samples...")
                model.eval()
                with torch.no_grad():
                    samples = sample_fn(
                        model.module if hasattr(model, 'module') else model,
                        tag_processor, inputs.shape[-1], 16, fixed_prompts, device,
                        cfg_scale=config['training'].get('cfg_scale', 1.4),
                        noise=fixed_noise
                    )

                    if use_tiny_vae:
                        samples = samples.to(dtype=torch.bfloat16)

                        # Reverse the FLUX.2 normalization
                        latents = (samples * latents_std) + latents_mean

                        with torch.no_grad():
                            # Decode directly (no pixel shuffle needed)
                            out = vae.decode(latents, return_dict=False)
                            recon = out[0] if isinstance(out, tuple) else out

                            # Convert from [-1, 1] to [0, 1] for saving/logging
                            samples = recon.clamp(-1, 1) / 2.0 + 0.5
                            samples = samples.to(dtype=torch.float32)

                    elif use_vae:
                        samples = samples.to(dtype=torch.bfloat16)

                        # Shuffle: 128 channels -> 32 channels (Height and Width are doubled)
                        latents = F.pixel_shuffle(samples, 2)

                        # Reverse the FLUX.2 normalization
                        latents = (latents * latents_std) + latents_mean

                        with torch.no_grad():
                            # Decode back to pixel space
                            recon = vae.decode(latents).sample

                            # Convert from [-1, 1] to [0, 1] for saving/logging
                            samples = recon.clamp(-1, 1) / 2.0 + 0.5
                            samples = samples.to(dtype=torch.float32)

                    grid = make_grid(samples, nrow=4)
                    wandb.log({"val/samples": wandb.Image(grid, caption=f"Step {global_step}")}, step=global_step)
                model.train()

    if rank == 0:
        save_checkpoint(model, optimizer, rank, config['training']['output_dir'],
                        global_step, config, fixed_prompts, fixed_noise)
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)