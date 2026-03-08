import torch
import torch.nn as nn
from pytorch_optimizer.optimizer import ScheduleFreeAdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
import os
import argparse
import numpy as np
from tqdm.auto import tqdm
import wandb
import glob
import builtins
import huggingface_hub
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import CoAtNeXtEncoder, SIGReg
from dataset import WDSLoader


def info_nce_loss(z1, z2, temperature=0.07):
    """Symmetric InfoNCE (SimCLR-style) contrastive loss.

    Pulls positive pairs (z1[i], z2[i]) together and pushes
    all other cross-batch pairs apart in cosine space.

    Args:
        z1, z2: (N, D) decoded logits from the two views
        temperature: softmax temperature (lower = sharper, typical 0.07-0.2)
    Returns:
        scalar InfoNCE loss
    """
    N = z1.size(0)
    # L2-normalize so dot product = cosine similarity
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Concatenate both views: (2N, D)
    z = torch.cat([z1, z2], dim=0)

    # Full (2N x 2N) cosine similarity matrix, scaled by temperature
    sim = (z @ z.T) / temperature

    # Mask out self-similarities on the diagonal
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)

    # Labels: for sample i in z1, the positive is at index N+i (in z2), and vice versa
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),  # z1[i]'s positive is z2[i] at idx N+i
        torch.arange(0, N, device=z.device),       # z2[i]'s positive is z1[i] at idx i
    ])

    return F.cross_entropy(sim, labels)


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
    # Find all checkpoints
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    # Sort by step number (extracted from filename)
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Remove older checkpoints if we have more than max_checkpoints
    if len(checkpoints) > max_checkpoints:
        checkpoints_to_remove = checkpoints[:-max_checkpoints]
        for ckpt in checkpoints_to_remove:
            try:
                os.remove(ckpt)
                print(f"Removed old checkpoint: {ckpt}")
            except OSError as e:
                print(f"Error removing {ckpt}: {e}")


def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    # Suppress printing on non-master ranks
    if rank != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Using device: {device}, Rank: {rank}, World Size: {world_size}")

    # Initialize WandB only on rank 0
    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-SSL'), config=config)

    # Load Data (Direct RGB returns, no need for VAE compression target coords)
    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'],
        tags_path=config['data']['tags_path'],
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    dataloader = wds_loader.make_loader()

    # Init CoAtNeXt Pure SSL Model
    num_tags = wds_loader.num_tags
    model = CoAtNeXtEncoder(
        backbone_model=config['model'].get('backbone_model', 'coatnext_nano_rw_224.sw_in1k'),
        proj_dim=config['model'].get('proj_dim', 8192),
        pretrained=True
    ).to(device)

    print(f"Initialized Pure LeJEPA SSL Model: {config['model'].get('backbone_model', 'coatnext_nano_rw_224.sw_in1k')}")
    print(f"Using Projection Dim: {config['model'].get('proj_dim', 8192)}")

    optimizer = ScheduleFreeAdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-2, betas=(0.9, 0.95))

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    resume_path = config['training'].get('resume_from', "")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "global_step" in checkpoint:
                    global_step = checkpoint["global_step"]
            else:
                model.load_state_dict(checkpoint)

            print("Successfully loaded checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Wrap model in DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # model.compile() # Optional, can enable if needed
    os.makedirs(config['training']['output_dir'], exist_ok=True)

    lamb_lejepa = config['training'].get('lejepa_lambda', 0.02)
    l1_weight = config['training'].get('l1_weight', 1e-4)
    nce_lambda = config['training'].get('nce_lambda', 0.1)
    nce_temperature = config['training'].get('nce_temperature', 0.07)

    # Calculate max_train_steps if not explicitly provided
    max_train_steps = config['training'].get('max_train_steps', config['training']['num_epochs'] * 1000)

    # Create progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(range(global_step, max_train_steps), desc="Steps", dynamic_ncols=True)
    else:
        pbar = None

    data_iter = iter(dataloader)

    # Main Training Loop
    while global_step < max_train_steps:
        model.train()
        optimizer.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # We take advantage of the dual views added to dataset.py
        # Expected from WDSLoader: image1, coords1, image2, tags
        x1, _, x2, _ = batch
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Forward pass raw images
            bin1, logits1 = model(x1)
            bin2, logits2 = model(x2)

        # LeJEPA losses operate in float32 (SIGReg disables autocast internally)
        # Stack views: (2, N, proj_dim)
        proj = torch.stack([logits1.float(), logits2.float()], dim=0)

        # Invariance: each view should predict the mean of all views (official formulation)
        inv_loss = (proj.mean(0) - proj).square().mean()

        # Cosine similarity loss: angular alignment between the two views (scale-invariant)
        # Replaced by InfoNCE: pushes positives together AND negatives apart
        infonce_loss = info_nce_loss(logits1.float(), logits2.float(), temperature=nce_temperature)

        # Uniformity: SIGReg on the primary view's continuous projections
        sigreg_loss = SIGReg(logits1, global_step=global_step, num_slices=256, chunk_size=32)

        # Sparsity: L1 norm on the binary STE outputs
        l1_loss = l1_weight * torch.mean(torch.abs(bin1.float()))

        # Composite LeJEPA Total Loss
        loss = (lamb_lejepa * sigreg_loss) + ((1 - lamb_lejepa) * inv_loss) + nce_lambda * infonce_loss + l1_loss

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step += 1

        # Update progress bar and logs
        if rank == 0:
            pbar.update(1)

            current_lr = optimizer.param_groups[0]['lr']
            active_binary_tags = bin1.sum(dim=1).mean().item()
            
            logs = {
                "loss": loss.item(),
                "loss_lejepa_combined": (lamb_lejepa * sigreg_loss + (1 - lamb_lejepa) * inv_loss).item(),
                "loss_sigreg": sigreg_loss.item(),
                "loss_inv": inv_loss.item(),
                "loss_nce": infonce_loss.item(),
                "loss_l1": l1_loss.item(),
                "active_binary_tags": active_binary_tags,
                "lr": current_lr,
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            }
            
            pbar.set_postfix(**logs)

            # Log to W&B (only on rank 0)
            if global_step % config['training']['log_every_steps'] == 0:
                wandb_log = {f"train/{k}": v for k, v in logs.items()}
                wandb.log(wandb_log, step=global_step)

        # Save checkpoint (only on rank 0)
        if global_step % config['training'].get('save_every_steps', 1000) == 0:
            # Synchronize all processes before checkpoint
            if is_ddp:
                dist.barrier()

            if rank == 0:
                print("\nSaving Checkpoint...")

                # Save Checkpoint (Unwrap DDP)
                model_to_save = model.module if is_ddp else model
                
                ckpt_state = {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "config": config
                }
                ckpt_path = os.path.join(config['training']['output_dir'], f'ckpt_step_{global_step}.pth')
                torch.save(ckpt_state, ckpt_path)
                cleanup_checkpoints(config['training']['output_dir'], config.get('max_checkpoints', 3), rank)

                # --- Eval: 4×2 image grid with per-pair cosine similarity ---
                model.eval()
                optimizer.eval()
                with torch.no_grad():
                    n_pairs = 4
                    imgs1 = x1[:n_pairs].float()  # (4, 3, H, W) in [-1, 1]
                    imgs2 = x2[:n_pairs].float()
                    _, lgt1 = model(imgs1.to(device))
                    _, lgt2 = model(imgs2.to(device))
                    pair_cos = F.cosine_similarity(lgt1, lgt2, dim=1)  # (4,)

                    # Unnormalize images to [0, 1] for display
                    imgs1_vis = (imgs1 * 0.5 + 0.5).clamp(0, 1).cpu()
                    imgs2_vis = (imgs2 * 0.5 + 0.5).clamp(0, 1).cpu()

                    # Build 4×2 grid: each row is (x1, x2), left col = view1, right col = view2
                    fig, axes = plt.subplots(4, 2, figsize=(5, 10))
                    for idx in range(n_pairs):
                        sim = pair_cos[idx].item()
                        img1_np = imgs1_vis[idx].permute(1, 2, 0).numpy()
                        img2_np = imgs2_vis[idx].permute(1, 2, 0).numpy()
                        axes[idx, 0].imshow(img1_np)
                        axes[idx, 0].set_title(f"View 1" if idx == 0 else "")
                        axes[idx, 0].axis('off')
                        axes[idx, 1].imshow(img2_np)
                        axes[idx, 1].set_title(f"View 2" if idx == 0 else "")
                        axes[idx, 1].axis('off')
                        # Annotate cosine sim on the right image
                        axes[idx, 1].set_xlabel(f"cos_sim={sim:.3f}", fontsize=9)
                    plt.suptitle(f"Step {global_step} | Avg cos_sim={pair_cos.mean().item():.3f}", fontsize=10)
                    plt.tight_layout()
                    wandb.log({"eval/view_pairs": wandb.Image(fig)}, step=global_step)
                    plt.close(fig)
                model.train()
                optimizer.train()
                
                # Upload to HuggingFace every 2 saves
                save_count = global_step // config['training']['save_every_steps']
                if save_count % 2 == 0 and save_count > 0:
                    print("Uploading checkpoints to Hugging Face Hub (Shio-Koube/Pure-CoAtNeXt-SSL)...")
                    try:
                        api = huggingface_hub.HfApi()
                        api.create_repo(repo_id="Shio-Koube/Pure-CoAtNeXt-SSL", exist_ok=True, repo_type="model")
                        api.upload_folder(
                            folder_path=config['training']['output_dir'],
                            repo_id="Shio-Koube/Pure-CoAtNeXt-SSL",
                            repo_type="model",
                            commit_message=f"Upload checkpoint step {global_step}"
                        )
                        print("Upload successful!")
                    except Exception as e:
                        print(f"Failed to upload to Hugging Face Hub: {e}")

            # Synchronize again after checkpoint
            if is_ddp:
                dist.barrier()

    print("Training Complete.")
    if rank == 0:
        pbar.close()
        final_path = os.path.join(config['training']['output_dir'], 'coatnext_ssl_final.pth')
        model_to_save = model.module if is_ddp else model
        torch.save(model_to_save.state_dict(), final_path)
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)