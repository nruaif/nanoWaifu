import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
import os
import argparse
import numpy as np
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
import glob
import builtins
import random
from dataset import WDSLoader
from torch.optim import AdamW
import torch.nn.functional as F
import bitsandbytes as bnb

# Dynamic model import
def get_model_and_sampler(config):
    if config['model'].get('use_v2_model', False):
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

def save_checkpoint(model, optimizer, rank, output_dir, step, config, fixed_prompts=None, fixed_noise=None):
    if rank != 0: return
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
    cleanup_checkpoints(output_dir, config.get('max_checkpoints', 3), rank)
    print(f"Checkpoint saved: {ckpt_path}")

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
    in_channels = config['model'].get('in_channels', 3)
    if use_vae:
        from flux2_tiny_autoencoder import Flux2TinyAutoEncoder
        print(">>> Loading FLUX.2-Tiny-AutoEncoder...")
        tiny_vae = Flux2TinyAutoEncoder.from_pretrained(
            "fal/FLUX.2-Tiny-AutoEncoder",
        ).to(device=device, dtype=torch.bfloat16).eval()
        # VAE output is 128 channels. Reshuffled to 32 channels.
        in_channels = 32
        print(f">>> VAE Mode Enabled: Model in_channels adjusted to {in_channels}")

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
    resume_path = config.get('resume_from', "outputs/")
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

            # Filter mismatching channels if transitioning to VAE or vice versa
            state_dict = checkpoint["model_state_dict"]
            model_state = model_to_load.state_dict()
            for k in ["conv_in.weight", "conv_in.bias", "conv_out.weight", "conv_out.bias"]:
                if k in state_dict and state_dict[k].shape != model_state[k].shape:
                    print(f">>> Channel Mismatch: Removing {k} from state_dict and re-initializing.")
                    del state_dict[k]

            model_to_load.load_state_dict(state_dict, strict=False)
            global_step = checkpoint["global_step"]
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

            # VAE Encoding if enabled
            if use_vae:
                with torch.inference_mode():
                    # Flux Tiny VAE takes images in [-1, 1]
                    v_images = images * 2.0 - 1.0
                    v_images = v_images.to(dtype=torch.bfloat16)
                    latents = tiny_vae.encode(v_images, return_dict=False)
                    # Scale 0.62, Shift 0
                    latents = latents * 0.62
                    # Reshuffle: 128 -> 32 channels (factor 2)
                    inputs = F.pixel_shuffle(latents, 2).to(dtype=torch.float32)
            else:
                inputs = images

            if rank == 0 and fixed_prompts is None:
                # Capture initial validation samples
                fixed_prompts = prompts[:16]
                fixed_noise = torch.randn_like(inputs[:16])

            # Flow Matching / Rectified Flow Training
            t = torch.rand((inputs.shape[0],), device=device)
            t_reshaped = t.view(-1, 1, 1, 1)
            noise = torch.randn_like(inputs)
            xt = (1 - t_reshaped) * inputs + t_reshaped * noise

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(xt, t, y_indices, y_offsets)
                # Scale loss by accumulation steps
                loss = F.mse_loss(pred, inputs) / accum_steps

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
                wandb.log({"train/loss": avg_loss}, step=global_step)
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

                    if use_vae:
                        # Reshuffle: 32 -> 128 channels (factor 2)
                        latents = F.pixel_unshuffle(samples, 2).to(dtype=torch.bfloat16)
                        # Inverse Scale 0.62, Shift 0
                        latents = latents / 0.62
                        with torch.inference_mode():
                            recon = tiny_vae.decode(latents, return_dict=False)
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
