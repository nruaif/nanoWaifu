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
from model import BinaryAutoencoder, PatchDiscriminator, disc_hinge_loss, gen_hinge_loss, r1_gradient_penalty
from dataset import WDSLoader
from config import Config


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


def get_pca_latent(latent):
    B, C, H, W = latent.shape
    X = latent.permute(0, 2, 3, 1).reshape(-1, C)
    X = X - X.mean(dim=0)
    U, S, V = torch.pca_lowrank(X, q=3)
    X_pca = torch.matmul(X, V[:, :3])
    X_pca = (X_pca - X_pca.min()) / (X_pca.max() - X_pca.min() + 1e-5)
    return X_pca.reshape(B, H, W, 3).permute(0, 3, 1, 2)


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

    # ==================== Models ====================
    model = BinaryAutoencoder(
        dims=cfg.model.dims,
        depths=cfg.model.depths,
        latent_discrete=cfg.model.latent_discrete,
        num_transformer_blocks=cfg.model.num_transformer_blocks,
        num_cls_tokens=cfg.model.num_cls_tokens,
        use_masking=cfg.training.use_masking,
        mask_block_size=cfg.training.mask_block_size,
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
            model_raw = model.module if is_ddp else model
            model_raw.anneal_temperature(factor=cfg.training.temp_anneal_factor, min_temp=cfg.training.temp_anneal_min)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x_real = batch[0].to(device, memory_format=torch.channels_last)

        # During disc warmup: freeze autoencoder, only train discriminator
        ae_frozen = cfg.gan.enabled and (global_step < cfg.gan.disc_warmup_steps)

        # ==================== Generator Step ====================
        if not ae_frozen:
            opt_gen.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            x_rec, latent, masked_ids = model(x_real)
            recon_loss = charbonnier(x_rec, x_real)
            perc_loss = lpips_fn(x_rec, x_real).mean()

            # Generator adversarial loss (only after warmup)
            g_adv_loss = torch.tensor(0.0, device=device)
            if cfg.gan.enabled and not ae_frozen:
                fake_preds = disc(x_rec)
                g_adv_loss = gen_hinge_loss(fake_preds)

            g_total = recon_loss + cfg.training.perceptual_weight * perc_loss
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
                fake_preds = disc(x_rec.detach())
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

            model_raw = model.module if is_ddp else model
            current_temp = model_raw.quant.temp.item()

            with torch.no_grad():
                B = x_real.shape[0]
                if masked_ids is not None and len(masked_ids) > 0:
                    is_masked = torch.zeros(B, dtype=torch.bool, device=x_real.device)
                    is_masked[masked_ids] = True
                    psnr_clean = compute_psnr(x_rec[~is_masked], x_real[~is_masked])
                    psnr_masked = compute_psnr(x_rec[is_masked], x_real[is_masked])
                else:
                    psnr_clean = compute_psnr(x_rec, x_real)
                    psnr_masked = 0.0

            postfix = {
                "rec": f"{recon_loss.item():.4f}",
                "perc": f"{perc_loss.item():.4f}",
                "psnr_c": f"{psnr_clean:.1f}",
                "psnr_m": f"{psnr_masked:.1f}",
                "temp": f"{current_temp:.4f}",
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
                    "train/total_loss": g_total.item(),
                    "train/lr": opt_gen.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/temp": current_temp,
                    "train/psnr_clean": psnr_clean,
                    "train/psnr_masked": psnr_masked,
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

                    # Clean reconstruction (eval mode = no masking)
                    x_rec_clean, _, _ = model(x1_sample)
                    psnr_clean_viz = compute_psnr(x_rec_clean, x1_sample)

                    clean_grid = make_grid(
                        torch.cat([x1_sample, x_rec_clean], dim=0),
                        nrow=n_viz, normalize=True, value_range=(-1, 1)
                    )
                    wandb.log({
                        "eval/clean_recon": wandb.Image(
                            clean_grid,
                            caption=f"Step {global_step} | Clean PSNR: {psnr_clean_viz:.2f} dB | Top: Orig, Bottom: Recon"
                        )
                    }, step=global_step)

                    # Masked reconstruction
                    model.train()
                    x_rec_masked, _, masked_viz_ids = model(x1_sample)
                    model.eval()

                    if masked_viz_ids is not None and len(masked_viz_ids) > 0:
                        masked_orig = x1_sample[masked_viz_ids]
                        masked_recon = x_rec_masked[masked_viz_ids]
                        psnr_masked_viz = compute_psnr(masked_recon, masked_orig)

                        masked_grid = make_grid(
                            torch.cat([masked_orig, masked_recon], dim=0),
                            nrow=masked_orig.shape[0], normalize=True, value_range=(-1, 1)
                        )
                        wandb.log({
                            "eval/masked_recon": wandb.Image(
                                masked_grid,
                                caption=f"Step {global_step} | Masked PSNR: {psnr_masked_viz:.2f} dB | Top: Orig, Bottom: Recon"
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
