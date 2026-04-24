import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_optimizer.optimizer import ScheduleFreeAdamW
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

from model import SimpleAutoencoder
from dataset import WDSLoader


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


def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

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

    model = SimpleAutoencoder(
        in_channels=config['model'].get('in_channels', 3),
        hidden_dims=config['model'].get('hidden_dims', [64, 128, 256, 512, 512]),
        expand_ratio=config['model'].get('expand_ratio', 4),
        num_transformer_blocks=config['model'].get('num_transformer_blocks', 4),
        dim=config['model'].get('dim', 16)
    ).to(device)

    optimizer = ScheduleFreeAdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-2)

    start_epoch = 0
    global_step = 0
    resume_path = config.get('resume_from', "")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                global_step = checkpoint.get("global_step", 0)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    os.makedirs(config['training']['output_dir'], exist_ok=True)

    max_train_steps = config['training'].get('max_train_steps', config['training']['num_epochs'] * 1000)

    if rank == 0:
        pbar = tqdm(range(global_step, max_train_steps), desc="Steps", dynamic_ncols=True)
    else:
        pbar = None

    data_iter = iter(dataloader)

    while global_step < max_train_steps:
        model.train()
        optimizer.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x1 = batch[0].to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            x_rec, latent = model(x1)
            loss = torch.mean((x_rec - x1) ** 2)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step += 1

        if rank == 0:
            pbar.update(1)
            logs = {
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            }
            pbar.set_postfix(**logs)

            if global_step % config['training']['log_every_steps'] == 0:
                wandb_log = {f"train/{k}": v for k, v in logs.items()}
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
                    "optimizer_state_dict": optimizer.state_dict()
                }
                ckpt_path = os.path.join(config['training']['output_dir'], f'ckpt_step_{global_step}.pth')
                torch.save(ckpt_state, ckpt_path)
                cleanup_checkpoints(config['training']['output_dir'], config.get('max_checkpoints', 3), rank)

                model.eval()
                optimizer.eval()
                with torch.no_grad():
                    # Visualize first 4 samples from batch
                    x1_sample = x1[:4]
                    x_rec, latent = model(x1_sample)

                    # Denormalize assuming x1 is in [-1, 1] range to [0, 1]
                    x1_vis = (x1_sample + 1) / 2.0
                    x_rec_vis = (x_rec + 1) / 2.0
                    
                    # Difference map
                    diff = torch.abs(x1_vis - x_rec_vis)
                    
                    # PCA of latent
                    latent_pca = get_pca_latent(latent)
                    # Interpolate PCA latent to match original image size for easier viewing
                    latent_pca = F.interpolate(latent_pca, size=(x1_sample.shape[2], x1_sample.shape[3]), mode='nearest')

                    # Concatenate to form a grid: row 1 (original), row 2 (recon), row 3 (diff), row 4 (pca)
                    all_vis = torch.cat([x1_vis, x_rec_vis, diff, latent_pca], dim=0)
                    all_vis = torch.clamp(all_vis, 0, 1)
                    
                    grid = make_grid(all_vis, nrow=4)
                    wandb_image = wandb.Image(grid, caption=f"Orig | Recon | Diff | PCA (Step {global_step})")
                    wandb.log({"samples": wandb_image}, step=global_step)

                model.train()
                optimizer.train()
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
