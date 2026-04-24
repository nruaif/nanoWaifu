import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.optim.lr_scheduler import LambdaLR
from model import BinaryAutoencoder
from dataset import WDSLoader


# ==========================================================
# Losses & Metrics
# ==========================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier loss: blend of L1 + L2 for sharper reconstructions than pure MSE."""
    def __init__(self, eps=5e-4):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.abs(diff).mean() + torch.square(diff).mean()


def compute_psnr(pred, target):
    """Compute PSNR for batch of images in [-1, 1] range."""
    # Convert to [0, 1] range
    pred = (pred + 1) / 2
    target = (target + 1) / 2

    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


# ==========================================================
# DDP Utilities
# ==========================================================

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


def get_pca_latent(latent):
    B, C, H, W = latent.shape
    X = latent.permute(0, 2, 3, 1).reshape(-1, C)
    X = X - X.mean(dim=0)
    U, S, V = torch.pca_lowrank(X, q=3)
    X_pca = torch.matmul(X, V[:, :3])
    # Normalize to [0, 1] for visualization
    X_pca = (X_pca - X_pca.min()) / (X_pca.max() - X_pca.min() + 1e-5)
    return X_pca.reshape(B, H, W, 3).permute(0, 3, 1, 2)


# ==========================================================
# Training
# ==========================================================

def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    # Performance settings
    torch.backends.cudnn.benchmark = True

    def lr_lambda(current_step):
        if current_step < 1000:
            return float(current_step) / float(max(1, 1000))
        return 1.0

    if rank != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Using device: {device}, Rank: {rank}, World Size: {world_size}")

    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-AE'), config=config)

    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'],
        csv_path=config['data']['csv_path'],
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    dataloader = wds_loader.make_loader()

    model = BinaryAutoencoder(
        dims=config['model'].get('dims', [96, 192, 384, 768]),
        depths=config['model'].get('depths', [3, 3, 9, 3]),
        latent_discrete=config['model'].get('latent_discrete', 256),
        latent_continuous=config['model'].get('latent_continuous', 32),
        residual_dropout_prob=config['training'].get('residual_dropout', 0.1),
        num_transformer_blocks=config['model'].get('num_transformer_blocks', 8),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # GradScaler for proper mixed-precision training
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    model.compile()
    start_epoch = 0
    global_step = 0

    # Loss function
    charbonnier = CharbonnierLoss()

    # Load checkpoint
    resume_path = config['training'].get('resume_from', "")
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                global_step = checkpoint.get("global_step", 0)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Successfully loaded checkpoint at step {global_step}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("No checkpoint found, starting fresh.")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    scheduler = LambdaLR(optimizer, lr_lambda)
    os.makedirs(config['training']['output_dir'], exist_ok=True)

    max_train_steps = config['training'].get('max_train_steps', config['training']['num_epochs'] * 1000)
    temp_anneal_factor = config['training'].get('temp_anneal_factor', 0.98)
    temp_anneal_min = config['training'].get('temp_anneal_min', 0.01)

    if rank == 0:
        pbar = tqdm(range(global_step, max_train_steps), desc="Steps", dynamic_ncols=True)
    else:
        pbar = None

    data_iter = iter(dataloader)

    while global_step < max_train_steps:
        model.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            # Epoch boundary: anneal temperature
            model_raw = model.module if is_ddp else model
            model_raw.anneal_temperature(factor=temp_anneal_factor, min_temp=temp_anneal_min)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x1 = batch[0].to(device, memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            x_rec, latent = model(x1)
            loss = charbonnier(x_rec, x1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        if rank == 0:
            pbar.update(1)

            # Get current temperature
            model_raw = model.module if is_ddp else model
            current_temp = model_raw.encoder.quant.temp.item()

            logs = {
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "temp": current_temp,
            }
            pbar.set_postfix(**{k: f"{v:.4f}" for k, v in logs.items()})

            if global_step % config['training']['log_every_steps'] == 0:
                # Compute PSNR
                with torch.no_grad():
                    psnr = compute_psnr(x_rec, x1)

                wandb_log = {f"train/{k}": v for k, v in logs.items()}
                wandb_log["train/psnr"] = psnr
                wandb.log(wandb_log, step=global_step)

        if global_step % config['training']['save_image_every_steps'] == 0:
            if is_ddp:
                dist.barrier()

            if rank == 0:
                print("\nSampling and Saving Checkpoint...")
                model_to_save = model.module if is_ddp else model
                ckpt_state = {
                    "model_state_dict": model_to_save.state_dict(),
                    "global_step": global_step,
                    "config": config,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }
                ckpt_path = os.path.join(config['training']['output_dir'], f'ckpt_step_{global_step}.pth')
                torch.save(ckpt_state, ckpt_path)
                cleanup_checkpoints(config['training']['output_dir'], config['training'].get('max_checkpoints', 3), rank)

                model.eval()
                with torch.no_grad():
                    n_viz = min(8, x1.shape[0])
                    x1_sample = x1[:n_viz]
                    x_rec_sample, _ = model(x1_sample)

                    # Compute PSNR for visualization batch
                    viz_psnr = compute_psnr(x_rec_sample, x1_sample)

                    # Top row: originals, Bottom row: reconstructions
                    comparison = torch.cat([x1_sample, x_rec_sample], dim=0)
                    grid = make_grid(comparison, nrow=n_viz, normalize=True, value_range=(-1, 1))

                    wandb_image = wandb.Image(
                        grid,
                        caption=f"Step {global_step} | PSNR: {viz_psnr:.2f} dB | Top: Original, Bottom: Recon"
                    )
                    wandb.log({"eval/reconstructions": wandb_image}, step=global_step)

                model.train()
                print("Checkpoint and sampling complete.\n")

            if is_ddp:
                dist.barrier()

    print("Training Complete.")
    if rank == 0:
        if pbar is not None:
            pbar.close()
        final_path = os.path.join(config['training']['output_dir'], 'ae_model_final.pth')
        model_to_save = model.module if is_ddp else model
        torch.save(model_to_save.state_dict(), final_path)
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)
