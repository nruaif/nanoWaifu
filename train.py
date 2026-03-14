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
from model import FCDM, TagProcessor, sample_flow
from dataset import WDSLoader
from torch.optim import AdamW
import torch.nn.functional as F
import bitsandbytes as bnb


# ---------------------------------------------------------------------------
# Helper: extract & normalize features
# ---------------------------------------------------------------------------


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


def save_checkpoint(model, optimizers, rank, output_dir, step, config, fixed_prompts=None, fixed_noise=None):
    if rank != 0: return
    print(f"\nSaving Checkpoint at step {step}...")
    model_to_save = model.module if hasattr(model, 'module') else model
    ckpt_path = os.path.join(output_dir, f'ckpt_step_{step}.pth')

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_muon_state_dict": optimizers[0].state_dict(),
        "optimizer_adamw_state_dict": optimizers[1].state_dict(),
        "global_step": step,
        "config": config,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "fixed_prompts": fixed_prompts,
        "fixed_noise": fixed_noise
    }

    torch.save(checkpoint, ckpt_path)
    cleanup_checkpoints(output_dir, config.get('max_checkpoints', 3), rank)
    print(f"Checkpoint saved to {ckpt_path}")


def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    if rank != 0:
        def print_pass(*args, **kwargs): pass
        builtins.print = print_pass

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Using device: {device}, Rank: {rank}, World Size: {world_size}")

    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-T2I'), config=config)

    # Tag Processor
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

    model = FCDM(
        in_channels=3,
        base_channels=config['model'].get('fcdm_dim', 128),
        num_blocks=config['model'].get('fcdm_depth', 2),
        num_classes=num_classes,
        patch_size=config['model'].get('patch_size', 16),
        use_t_cond=False
    ).to(device, memory_format=torch.channels_last)

    # Teacher model (EMA)
    model_teacher = FCDM(
        in_channels=3,
        base_channels=config['model'].get('fcdm_dim', 128),
        num_blocks=config['model'].get('fcdm_depth', 2),
        num_classes=num_classes,
        patch_size=config['model'].get('patch_size', 16),
        use_t_cond=False
    ).to(device, memory_format=torch.channels_last)
    model_teacher.load_state_dict(model.state_dict())
    for p in model_teacher.parameters():
        p.requires_grad = False
    
    # Projector for Lrep (Mid block student [4C] to Mid block teacher [4C])
    # Mid block is stage 3, channel count is model.c * 4
    projector = nn.Conv2d(model.c * 4, model.c * 4, kernel_size=1).to(device)

    # Optimizer setup (Muon for 2D params, AdamW for others)
    params_2d = []
    params_1d = []
    for name, param in list(model.named_parameters()) + list(projector.named_parameters()):
        if param.requires_grad:
            meaningful_dims = sum(1 for s in param.shape if s > 1)
            if meaningful_dims == 2:
                params_2d.append(param)
            else:
                params_1d.append(param)

    # Fallback to AdamW if Muon is not available or appropriate
    try:
        optimizer_muon = torch.optim.Muon(
            params_2d, lr=config['training']['learning_rate'],
            momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"
        )
    except (AttributeError, NameError):
        print("Muon optimizer not found, using AdamW for all parameters.")
        optimizer_muon = bnb.optim.AdamW8bit(params_2d, lr=config['training']['learning_rate'])

    optimizer_adamw = bnb.optim.AdamW8bit(
        params_1d, lr=config['training']['learning_rate'],
        weight_decay=0.1, betas=(0.9, 0.95),
    )

    optimizers = [optimizer_muon, optimizer_adamw]

    # --- RESUME LOGIC ---
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
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model_teacher.load_state_dict(checkpoint["model_state_dict"], strict=False)
            global_step = checkpoint["global_step"]
            if "fixed_noise" in checkpoint and checkpoint["fixed_noise"] is not None:
                fixed_noise = checkpoint["fixed_noise"].to(device)
            print(f"Successfully resumed at step {global_step}")

    if is_ddp:
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    os.makedirs(config['training']['output_dir'], exist_ok=True)
    cfg_scale = config['training'].get('cfg_scale', 4.0)
    max_train_steps = config['training'].get('max_train_steps', 1000000)
    ema_decay = config['training'].get('ema_decay', 0.999)
    patch_size = config['model'].get('patch_size', 16)
    
    # Calculate index of the last mid block
    # Stage 1: l blocks, Stage 2: 2l blocks, Stage 3: 4l blocks
    m_raw = model.module if hasattr(model, 'module') else model
    mid_idx = m_raw.l + (m_raw.l * 2) + (m_raw.l * 4)

    if rank == 0:
        pbar = tqdm(range(global_step, max_train_steps), desc="Steps", dynamic_ncols=True)
        running_metrics = {}
    else:
        pbar = None

    data_iter = iter(dataloader)

    while global_step < max_train_steps:
        model.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Batch is now (images, prompts, coords) where images is (B, C, H, W)
        images, prompts, _ = batch
        images = images.to(device, memory_format=torch.channels_last)
        
        # Classifier-free guidance dropout handled in TagProcessor
        dropout_prob = config['training'].get('class_dropout_prob', 0.1)
        y_indices, y_offsets = tag_processor.process_prompts(prompts, device, dropout_prob=dropout_prob)

        if rank == 0 and fixed_prompts is None:
            general_candidates = [p for p in prompts if "general" in p]
            if len(general_candidates) >= 16:
                fixed_prompts = general_candidates[:16]
                fixed_noise = torch.randn_like(images[:16])
                print(f"\n[Step {global_step}] Locked 16 'general' prompts for fixed validation.")

        # Dual-Timestep Scheduling
        t = torch.rand((images.shape[0],), device=device)
        s = torch.rand((images.shape[0],), device=device)
        
        # Patch-wise token mask
        h_p, w_p = images.shape[2] // patch_size, images.shape[3] // patch_size
        mask = (torch.rand((images.shape[0], 1, h_p, w_p), device=device) < 0.5).float()
        mask_up = F.interpolate(mask, size=(images.shape[2], images.shape[3]), mode='nearest')
        
        t_reshaped = t.view(-1, 1, 1, 1)
        s_reshaped = s.view(-1, 1, 1, 1)
        
        # Tau: mix of t and s based on mask
        tau = mask_up * s_reshaped + (1 - mask_up) * t_reshaped
        
        noise = torch.randn_like(images)
        # Student Input (heterogeneous noise)
        x_tau = (1 - tau) * images + tau * noise
        
        # Teacher Input (uniform less noisy input)
        tau_min = torch.min(t, s)
        tau_min_reshaped = tau_min.view(-1, 1, 1, 1)
        x_tau_min = (1 - tau_min_reshaped) * images + tau_min_reshaped * noise

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            # Student forward with feature extraction from last Mid block
            student_out, student_feats = model(x_tau, t, y_indices, y_offsets, feat_layers=[mid_idx])
            
            # Teacher forward with feature extraction from last Mid block
            with torch.no_grad():
                teacher_out, teacher_feats = model_teacher(x_tau_min, tau_min, y_indices, y_offsets, feat_layers=[mid_idx])
            
            # Extract features
            f_s = student_feats[mid_idx] # (B, 4C, 8, 8)
            f_t = teacher_feats[mid_idx] # (B, 4C, 8, 8)
            
            # Project student feature to match teacher
            #f_s_proj = projector(f_s)
            
            # Normalize for cosine similarity
            f_s_proj = F.normalize(f_s, dim=1)
            f_t = F.normalize(f_t, dim=1)
            
            # Lrep = 1 - cos(h, f) to keep it positive
            loss_rep = 1 - (f_s_proj * f_t).sum(dim=1).mean()
            
            # Standard Diffusion loss (MSE)
            loss_mse = F.mse_loss(student_out, images) + F.l1_loss(teacher_out, images)
            
            # Total loss
            loss = loss_mse + loss_rep

        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_muon.step()
        optimizer_adamw.step()

        # Teacher EMA Update
        with torch.no_grad():
            m_student = model.module if hasattr(model, 'module') else model
            for p_s, p_t in zip(m_student.parameters(), model_teacher.parameters()):
                p_t.data.mul_(ema_decay).add_(p_s.data, alpha=1 - ema_decay)

        global_step += 1

        if rank == 0:
            pbar.update(1)
            logs = {
                "loss": loss.item(),
                "loss_mse": loss_mse.item(),
                "loss_rep": loss_rep.item(),
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            }
            
            for k, v in logs.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + float(v)
            
            pbar.set_postfix(**logs)
            if global_step % config['training']['log_every_steps'] == 0:
                avg_logs = {f"train/{k}": v / config['training']['log_every_steps'] for k, v in running_metrics.items()}
                wandb.log(avg_logs, step=global_step)
                running_metrics = {}
        if global_step % config['training']['save_image_every_steps'] == 0:
            if is_ddp: dist.barrier()
            if rank == 0:
                save_checkpoint(model, optimizers, rank, config['training']['output_dir'],
                                global_step, config, fixed_prompts, fixed_noise)

                print("\nSampling...")
                model.eval()
                with torch.no_grad():
                    use_prompts = fixed_prompts if fixed_prompts is not None else prompts[:16]
                    use_noise = fixed_noise if fixed_noise is not None else None
                    
                    samples = sample_flow(
                        model.module if hasattr(model, 'module') else model, 
                        tag_processor, image_size, len(use_prompts),
                        use_prompts, device,
                        cfg_scale=cfg_scale,
                        noise=use_noise
                    )
                    grid = make_grid(samples, nrow=4)
                    wandb.log({"samples": wandb.Image(grid, caption=f"Step {global_step}: {use_prompts[0]}...")},
                              step=global_step)
                model.train()
            if is_ddp: dist.barrier()

    if rank == 0:
        save_checkpoint(model, optimizers, rank, config['training']['output_dir'],
                        global_step, config, fixed_prompts, fixed_noise)
        pbar.close()
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    train(args.config)
