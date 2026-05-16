import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
import numpy as np
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
import glob
import builtins
import lpips
from torch.optim.lr_scheduler import LambdaLR
from sphere_encoder import SphereAutoencoder
from model import PatchDiscriminator, disc_hinge_loss, gen_hinge_loss, r1_gradient_penalty
from dataset import WDSLoader
from config import Config


# ==========================================================
# Tag Processor
# ==========================================================

class TagProcessor:
    def __init__(self, class_map, max_tags=32):
        self.tag_to_idx = class_map
        if class_map:
            self.num_classes = max(class_map.values()) + 2
        else:
            self.num_classes = 1
        self.pad_idx = self.num_classes - 1
        self.max_tags = max_tags

    def process_prompts(self, prompts, device):
        batch_indices = []
        for p in prompts:
            tags = p.split()
            indices = []
            for t in tags:
                if t in self.tag_to_idx:
                    indices.append(self.tag_to_idx[t])
            indices = indices[:self.max_tags]
            indices += [self.pad_idx] * (self.max_tags - len(indices))
            batch_indices.append(indices)
        return torch.tensor(batch_indices, dtype=torch.long, device=device)


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


# ==========================================================
# Training
# ==========================================================

def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    torch.backends.cudnn.benchmark = True

    if rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    cfg = Config.from_yaml(config_path)

    print(f"Using device: {device}, Rank: {rank}, World Size: {world_size}")

    if rank == 0:
        wandb.init(project=cfg.training.wandb_project, config=cfg.to_dict())

    wds_loader = WDSLoader(
        url=cfg.data.webdataset_url,
        csv_path=cfg.data.csv_path,
        image_size=cfg.training.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    dataloader = wds_loader.make_loader()
    
    tag_processor = TagProcessor(wds_loader.class_map, max_tags=32)

    # ==================== Models ====================
    # Initialize SphereAutoencoder instead of BinaryAutoencoder
    model = SphereAutoencoder(
        vocab_size=tag_processor.num_classes,
        patch_size=cfg.model.patch_size,
        dim=cfg.model.dim,
        enc_depth=cfg.model.enc_depth,
        dec_depth=cfg.model.dec_depth,
        latent_dim=cfg.model.latent_dim,
        num_heads=cfg.model.num_heads
    ).to(device)

    # ==================== Discriminator ====================
    disc = None
    opt_disc = None
    scaler_disc = None

    if cfg.gan.enabled:
        disc = PatchDiscriminator(
            in_channels=3,
            ndf=cfg.gan.ndf,
            n_layers=cfg.gan.n_layers,
            num_scales=cfg.gan.num_scales,
        ).to(device)

        opt_disc = torch.optim.AdamW(
            disc.parameters(),
            lr=cfg.gan.disc_lr,
            weight_decay=0.0,
            betas=(0.0, 0.99),
        )
        scaler_disc = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
        print(f"GAN enabled: warmup={cfg.gan.disc_warmup_steps}, adv_weight={cfg.gan.adv_weight}, "
              f"r1_gamma={cfg.gan.r1_gamma}, r1_every={cfg.gan.r1_every}")

    # ==================== Generator Optimizer ====================
    opt_gen = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    global_step = 0

    # ==================== Losses ====================
    charbonnier = CharbonnierLoss()
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # ==================== Load Checkpoint ====================
    if cfg.training.resume_from and os.path.exists(cfg.training.resume_from):
        print(f"Resuming from checkpoint: {cfg.training.resume_from}")
        try:
            checkpoint = torch.load(cfg.training.resume_from, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                if "optimizer_state_dict" in checkpoint:
                    opt_gen.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                global_step = checkpoint.get("global_step", 0)
                if cfg.gan.enabled and "disc_state_dict" in checkpoint:
                    disc.load_state_dict(checkpoint["disc_state_dict"], strict=False)
                if cfg.gan.enabled and "opt_disc_state_dict" in checkpoint:
                    opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Successfully loaded checkpoint at step {global_step}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("No checkpoint found, starting fresh.")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        if cfg.gan.enabled:
            disc = DDP(disc, device_ids=[local_rank])

    # LR scheduler with warmup
    def lr_lambda(step):
        if step < cfg.training.warmup_steps:
            return float(step) / float(max(1, cfg.training.warmup_steps))
        return 1.0

    scheduler_gen = LambdaLR(opt_gen, lr_lambda)
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    if rank == 0:
        pbar = tqdm(range(global_step, cfg.training.max_train_steps), desc="Steps", dynamic_ncols=True)
    else:
        pbar = None

    data_iter = iter(dataloader)

    # ==================== Training Loop ====================
    while global_step < cfg.training.max_train_steps:
        model.train()
        if cfg.gan.enabled:
            disc.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x_real = batch[0].to(device, memory_format=torch.channels_last)
        prompts = batch[1]
        tags = tag_processor.process_prompts(prompts, device)

        # During disc warmup: freeze autoencoder, only train discriminator
        ae_frozen = cfg.gan.enabled and (global_step < cfg.gan.disc_warmup_steps)

        # ==================== Generator Step ====================
        if not ae_frozen:
            opt_gen.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            # Forward pass: returns noisy (small jitter), NOISY (large jitter), clean latent v, and conditional embed
            recon_noisy, recon_NOISY, v, cond = model(x_real, tags)
            
            # 1. Pixel Reconstruction Loss (small jitter to x_real)
            recon_loss = charbonnier(recon_noisy, x_real)
            perc_loss = lpips_fn(recon_noisy, x_real).mean()
            loss_pix_recon = recon_loss + cfg.training.perceptual_weight * perc_loss

            # 2. Pixel Consistency Loss (large jitter to small jitter output)
            recon_con_loss = charbonnier(recon_NOISY, recon_noisy.detach())
            perc_con_loss = lpips_fn(recon_NOISY, recon_noisy.detach()).mean()
            loss_pix_con = recon_con_loss + cfg.training.perceptual_weight * perc_con_loss
            
            # 3. Latent Consistency Loss (similarity of cleanly encoded large jitter vs clean latent)
            model_raw = model.module if is_ddp else model
            z_reenc, _, _ = model_raw.encoder(recon_NOISY, cond)
            loss_lat_con = 1.0 - F.cosine_similarity(v.flatten(1), z_reenc.flatten(1), dim=1).mean()

            # Generator adversarial loss (only after warmup)
            g_adv_loss = torch.tensor(0.0, device=device)
            if cfg.gan.enabled and not ae_frozen:
                fake_preds = disc(recon_noisy)
                g_adv_loss = gen_hinge_loss(fake_preds)

            # Combined losses
            g_total = loss_pix_recon + 1.0 * loss_pix_con + 0.1 * loss_lat_con
            
            if cfg.gan.enabled and not ae_frozen:
                g_total = g_total + cfg.gan.adv_weight * g_adv_loss

        if not ae_frozen:
            scaler.scale(g_total).backward()
            scaler.unscale_(opt_gen)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt_gen)
            scaler.update()
            scheduler_gen.step()
        else:
            g_total.backward()
            grad_norm = torch.tensor(0.0)

        # ==================== Discriminator Step ====================
        d_loss_val = 0.0
        r1_val = 0.0

        if cfg.gan.enabled:
            opt_disc.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                fake_preds = disc(recon_noisy.detach())
                real_preds = disc(x_real)
                d_loss = disc_hinge_loss(real_preds, fake_preds)

            scaler_disc.scale(d_loss).backward()

            # Lazy R1 gradient penalty
            if global_step % cfg.gan.r1_every == 0 and cfg.gan.r1_gamma > 0:
                x_real_r1 = x_real.detach().requires_grad_(True)
                real_preds_r1 = disc(x_real_r1)
                r1_penalty = r1_gradient_penalty(x_real_r1, real_preds_r1)
                r1_loss = (cfg.gan.r1_gamma / 2.0) * r1_penalty * cfg.gan.r1_every
                scaler_disc.scale(r1_loss).backward()
                r1_val = r1_penalty.item()

            scaler_disc.unscale_(opt_disc)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            scaler_disc.step(opt_disc)
            scaler_disc.update()

            d_loss_val = d_loss.item()

        global_step += 1

        # ==================== Logging ====================
        if rank == 0:
            pbar.update(1)

            with torch.no_grad():
                psnr_clean = compute_psnr(recon_noisy, x_real)

            postfix = {
                "rec": f"{recon_loss.item():.4f}",
                "perc": f"{perc_loss.item():.4f}",
                "p_con": f"{loss_pix_con.item():.4f}",
                "l_con": f"{loss_lat_con.item():.4f}",
                "psnr": f"{psnr_clean:.1f}",
            }
            if cfg.gan.enabled:
                postfix["d"] = f"{d_loss_val:.3f}"
                postfix["g_adv"] = f"{g_adv_loss.item():.3f}"
                if ae_frozen:
                    postfix["WARMUP"] = f"{cfg.gan.disc_warmup_steps - global_step}"
            pbar.set_postfix(**postfix)

            if global_step % cfg.training.log_every_steps == 0:
                wandb_log = {
                    "train/recon_loss": recon_loss.item(),
                    "train/perceptual_loss": perc_loss.item(),
                    "train/pix_con_loss": loss_pix_con.item(),
                    "train/lat_con_loss": loss_lat_con.item(),
                    "train/total_loss": g_total.item(),
                    "train/lr": opt_gen.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/psnr_clean": psnr_clean,
                }
                if cfg.gan.enabled:
                    wandb_log.update({
                        "train/d_loss": d_loss_val,
                        "train/g_adv_loss": g_adv_loss.item(),
                        "train/r1_penalty": r1_val,
                        "train/ae_frozen": float(ae_frozen),
                    })
                wandb.log(wandb_log, step=global_step)

        # ==================== Checkpointing & Visualization ====================
        if global_step % cfg.training.save_image_every_steps == 0:
            if is_ddp:
                dist.barrier()

            if rank == 0:
                print("\nSampling and Saving Checkpoint...")
                model_to_save = model.module if is_ddp else model
                ckpt_state = {
                    "model_state_dict": model_to_save.state_dict(),
                    "global_step": global_step,
                    "config": cfg.to_dict(),
                    "optimizer_state_dict": opt_gen.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }
                if cfg.gan.enabled:
                    disc_to_save = disc.module if is_ddp else disc
                    ckpt_state["disc_state_dict"] = disc_to_save.state_dict()
                    ckpt_state["opt_disc_state_dict"] = opt_disc.state_dict()
                    ckpt_state["scaler_disc_state_dict"] = scaler_disc.state_dict()

                ckpt_path = os.path.join(cfg.training.output_dir, f'ckpt_step_{global_step}.pth')
                torch.save(ckpt_state, ckpt_path)
                cleanup_checkpoints(cfg.training.output_dir, cfg.training.max_checkpoints, rank)

                model.eval()
                with torch.no_grad():
                    n_viz = min(8, x_real.shape[0])
                    x1_sample = x_real[:n_viz]
                    tags_sample = tags[:n_viz]

                    # Reconstruction 
                    x_rec_eval, _, _, _ = model(x1_sample, tags_sample, noise_r=0.0)
                    psnr_eval_viz = compute_psnr(x_rec_eval, x1_sample)

                    clean_grid = make_grid(
                        torch.cat([x1_sample, x_rec_eval], dim=0),
                        nrow=n_viz, normalize=True, value_range=(-1, 1)
                    )
                    wandb.log({
                        "eval/recon": wandb.Image(
                            clean_grid,
                            caption=f"Step {global_step} | PSNR: {psnr_eval_viz:.2f} dB | Top: Orig, Bottom: Recon"
                        )
                    }, step=global_step)
                    
                    # Generate from random noise
                    patch_size = model_to_save.patch_size
                    H_grid = x1_sample.shape[2] // patch_size
                    W_grid = x1_sample.shape[3] // patch_size
                    L = H_grid * W_grid
                    dim = model_to_save.encoder.to_latent.out_features
                    noise_e = torch.randn(n_viz, L, dim, device=device)
                    x_gen = model_to_save.generate(noise_e, H_grid, W_grid, tags=tags_sample, steps=2, cfg_scale=1.5)
                    gen_grid = make_grid(x_gen, nrow=n_viz, normalize=True, value_range=(-1, 1))
                    
                    wandb.log({
                        "eval/generation": wandb.Image(
                            gen_grid,
                            caption=f"Step {global_step} | 2-step Generation with CFG 1.5"
                        )
                    }, step=global_step)

                model.train()
                print("Checkpoint and sampling complete.\n")

            if is_ddp:
                dist.barrier()

    print("Training Complete.")
    if rank == 0:
        if pbar is not None:
            pbar.close()
        final_path = os.path.join(cfg.training.output_dir, 'ae_model_final.pth')
        model_to_save = model.module if is_ddp else model
        torch.save(model_to_save.state_dict(), final_path)
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)