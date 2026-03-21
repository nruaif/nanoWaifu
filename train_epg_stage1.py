import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import yaml
import os
import argparse
import numpy as np
from tqdm.auto import tqdm
import wandb
import builtins
import copy
import math
from dataset import WDSLoader
from model_epg import EPGEncoder, EPGProjector, EPGModel, TagProcessor
from torchvision import transforms
from siglip import SupConLoss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

torch.autograd.set_detect_anomaly(False)  # Disable in production — ~20% overhead


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, local_rank, world_size, device
    return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(target_model, source_model, beta):
    for tp, sp in zip(target_model.parameters(), source_model.parameters()):
        tp.data.mul_(beta).add_(sp.data, alpha=1 - beta)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def cosine_lr_lambda(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine  = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_confusion_matrix(sim_matrix: torch.Tensor, step: int, batch_size: int):
    """Logs Positive vs Negative similarity between view groups to detect collapse."""
    sim_matrix = sim_matrix.detach().float().contiguous()  # ensure contiguous before view
    k = sim_matrix.shape[0] // batch_size
    reshaped = sim_matrix.view(k, batch_size, k, batch_size)

    pos_sim = np.zeros((k, k))
    neg_sim = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            block = reshaped[i, :, j, :]                             # (B, B)
            pos_sim[i, j] = block.diagonal().mean().item()
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=block.device)
            neg_sim[i, j] = block[mask].mean().item()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pos_sim, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(im)

    views = ["Aug", "Noise", "Label"]
    ax.set_xticks(range(k)); ax.set_xticklabels(views)
    ax.set_yticks(range(k)); ax.set_yticklabels(views)

    for i in range(k):
        for j in range(k):
            color = "w" if pos_sim[i, j] < 0.5 else "k"
            ax.text(j, i, f"Pos: {pos_sim[i, j]:.2f}\nNeg: {neg_sim[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_title(f"Group Alignment: Positive vs Negative (Step {step})")
    wandb.log({"val/alignment_contrast": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(output_dir, prefix="epg_stage1_step_"):
    import re
    if not os.path.exists(output_dir):
        return None
    files = os.listdir(output_dir)
    ckpts = [f for f in files if f.startswith(prefix) and f.endswith(".pth")]
    if not ckpts:
        return None
    
    def get_step(f):
        match = re.search(r"step_(\d+)\.pth", f)
        return int(match.group(1)) if match else -1
        
    ckpts.sort(key=get_step)
    return os.path.join(output_dir, ckpts[-1])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class EPGStage1Trainer:
    def __init__(self, config, device, rank, world_size):
        self.config     = config
        self.device     = device
        self.rank       = rank
        self.world_size = world_size

        self.embed_dim   = config["model"].get("embed_dim", 768)
        self.proj_dim    = config["model"].get("proj_dim", 256)
        self.ema_beta    = config["training"].get("ema_beta", 0.99)
        self.num_classes = config["model"].get("num_classes", 12476)

        # Detect best autocast dtype
        self.amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # ---- Online networks ------------------------------------------------
        self.encoder   = EPGEncoder(
            embed_dim   = self.embed_dim,
            depth       = config["model"].get("depth", 12),
            num_heads   = config["model"].get("num_heads", 12),
            num_classes = self.num_classes,
        ).to(device)
        self.projector = EPGProjector(
            embed_dim=self.embed_dim, proj_dim=self.proj_dim
        ).to(device)

        if config["training"].get("gradient_checkpointing", False):
            self.encoder.gradient_checkpointing = True

        # Wrap in DDP
        if dist.is_initialized():
            self.encoder   = DDP(self.encoder,   device_ids=[device.index], find_unused_parameters=False)
            self.projector = DDP(self.projector, device_ids=[device.index], find_unused_parameters=False)

        # ---- EMA (target) networks ------------------------------------------
        # EMA networks are not wrapped in DDP — they stay on local device only
        raw_encoder   = self.encoder.module   if isinstance(self.encoder,   DDP) else self.encoder
        raw_projector = self.projector.module if isinstance(self.projector, DDP) else self.projector
        self.encoder_ema   = copy.deepcopy(raw_encoder).to(device)
        self.projector_ema = copy.deepcopy(raw_projector).to(device)
        for m in (self.encoder_ema, self.projector_ema):
            for p in m.parameters():
                p.requires_grad_(False)

        # ---- Optimizer ------------------------------------------------------
        params = (
            list(self.encoder.parameters()) +
            list(self.projector.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr           = config["training"]["learning_rate"],
            weight_decay = config["training"].get("weight_decay", 0.05),
        )

        # ---- LR scheduler (cosine with warmup) ------------------------------
        max_steps    = config["training"].get("max_train_steps", 600_000)
        warmup_steps = config["training"].get("warmup_steps", 2_000)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: cosine_lr_lambda(s, warmup_steps, max_steps),
        )

        # ---- Loss & AMP scaler ----------------------------------------------
        self.supcon = SupConLoss().to(device)
        if config["training"].get("compile", False):
            self.supcon = torch.compile(self.supcon)

        # GradScaler is a no-op when using bfloat16 (which doesn't need scaling)
        self.scaler = GradScaler(enabled=(self.amp_dtype == torch.float16))

        # ---- Augmentations --------------------------------------------------
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(config["training"]["image_size"], scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def load_checkpoint(self, ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        raw_encoder   = self.encoder.module   if isinstance(self.encoder,   DDP) else self.encoder
        raw_projector = self.projector.module if isinstance(self.projector, DDP) else self.projector
        
        raw_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        raw_projector.load_state_dict(checkpoint["projector_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Load EMA models if they exist in checkpoint
        if "encoder_ema_state_dict" in checkpoint:
            self.encoder_ema.load_state_dict(checkpoint["encoder_ema_state_dict"])
        else:
            # Fallback: init EMA from loaded online model
            self.encoder_ema.load_state_dict(raw_encoder.state_dict())
            
        if "projector_ema_state_dict" in checkpoint:
            self.projector_ema.load_state_dict(checkpoint["projector_ema_state_dict"])
        else:
            self.projector_ema.load_state_dict(raw_projector.state_dict())
            
        return checkpoint["global_step"]

    # ------------------------------------------------------------------
    def train_step(self, x, y_indices, y_offsets, y_labels):
        B = x.shape[0]
        self.encoder.train()
        self.projector.train()
        self.optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

        raw_encoder   = self.encoder.module   if isinstance(self.encoder,   DDP) else self.encoder
        raw_projector = self.projector.module if isinstance(self.projector, DDP) else self.projector

        with autocast("cuda", dtype=self.amp_dtype):
            # ---- Label embeddings (online + EMA) ----------------------------
            y_feat_online = self.projector(raw_encoder.get_y_feat(y_indices, y_offsets))
            with torch.no_grad():
                y_feat_target = self.projector_ema(
                    self.encoder_ema.get_y_feat(y_indices, y_offsets)
                )

            # ---- Augmented views --------------------------------------------
            y1, y2 = self.aug(x), self.aug(x)
            t0 = torch.zeros(B, device=self.device)

            # ---- Noise schedule (EDM-style) ----------------------------------
            N = 1280
            n      = torch.randint(1, N, (B,), device=self.device)
            tn     = n.float() / N
            tn_1   = (n - 1).float() / N
            tn_scaled   = 1000 * 0.25 * torch.log(tn.clamp(min=1e-8))
            tn_1_scaled = 1000 * 0.25 * torch.log(tn_1.clamp(min=1e-8))

            epsilon    = torch.randn_like(x)
            sigma_data = 0.5
            xtn   = (x + tn.view(-1, 1, 1, 1)   * epsilon) * \
                    (1.0 / torch.sqrt(tn  **2 + sigma_data**2)).view(-1, 1, 1, 1)
            xtn_1 = (x + tn_1.view(-1, 1, 1, 1) * epsilon) * \
                    (1.0 / torch.sqrt(tn_1**2 + sigma_data**2)).view(-1, 1, 1, 1)

            # ---- Online forward passes --------------------------------------
            feat_im1 = self.encoder(y1,  t0,         y_indices, y_offsets)
            q_im1    = self.projector(feat_im1[:, 0])
            feat_n   = self.encoder(xtn, tn_scaled,   y_indices, y_offsets)
            qn       = self.projector(feat_n[:, 0])

            # ---- EMA (target) forward passes (no grad) ----------------------
            with torch.no_grad():
                feat_im2 = self.encoder_ema(y2,   t0,          y_indices, y_offsets)
                q_im2    = self.projector_ema(feat_im2[:, 0])
                feat_n_1 = self.encoder_ema(xtn_1, tn_1_scaled, y_indices, y_offsets)
                qn_1     = self.projector_ema(feat_n_1[:, 0])

            loss, _, sim_matrix = self.supcon(
                [q_im1, qn, y_feat_online],
                [q_im2, qn_1, y_feat_target],
                labels=y_labels,
            )

        # ---- Backward + optimizer step --------------------------------------
        self.scaler.scale(loss).backward()

        if self.config["training"].get("grad_clip", 0.0) > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.projector.parameters()),
                self.config["training"]["grad_clip"],
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # EMA update uses raw (non-DDP) modules
        update_ema(self.encoder_ema,   raw_encoder,   self.ema_beta)
        update_ema(self.projector_ema, raw_projector, self.ema_beta)

        return {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, sim_matrix


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path: str):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    # Silence non-rank-0 processes
    if rank != 0:
        builtins.print = lambda *a, **kw: None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    dataloader = WDSLoader(
        url        = config["data"]["webdataset_url"],
        csv_path   = config["data"].get("csv_path"),
        image_size = config["training"]["image_size"],
        batch_size = config["training"]["batch_size"],
        num_workers= config["training"]["num_workers"],
    ).make_loader()

    tag_processor = TagProcessor("tags.txt")
    trainer       = EPGStage1Trainer(config, device, rank, world_size)
    max_steps     = config["training"].get("max_train_steps", 600_000)
    global_step   = 0

    # ---- Autoresume ---------------------------------------------------------
    latest_ckpt = find_latest_checkpoint(config["training"]["output_dir"])
    if latest_ckpt:
        global_step = trainer.load_checkpoint(latest_ckpt)
        print(f"Resuming from step {global_step}")

    if rank == 0:
        wandb.init(project="EPG-Stage1-SupCon", config=config, resume="allow", id=config["training"].get("wandb_id"))
        pbar = tqdm(total=max_steps, desc="EPG Stage 1")
        pbar.update(global_step)

    # ---- Training loop ------------------------------------------------------
    # Keep a single persistent iterator; recreate only on exhaustion
    data_iter = iter(dataloader)

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)  # reassign so iterator persists correctly
            batch     = next(data_iter)

        images, prompts, _ = batch

        # prompts are strings — pass them directly; don't call .to(device)
        y_indices, y_offsets, y_labels = tag_processor.process_prompts(prompts, device)

        avg_tags = len(y_indices) / images.shape[0]

        # Non-blocking transfer to GPU
        images = images.to(device, non_blocking=True)

        metrics, sim_matrix = trainer.train_step(
            images, y_indices, y_offsets, y_labels
        )
        metrics["avg_tags"] = avg_tags
        global_step += 1

        if rank == 0:
            pbar.update(1)

            log_every  = config["training"].get("log_every_steps",      100)
            save_every = config["training"].get("save_ckpt_every_steps", 5000)

            if global_step % log_every == 0:
                wandb.log(metrics, step=global_step)
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr":   f"{metrics['lr']:.2e}",
                    "tags": f"{avg_tags:.1f}",
                })

            if global_step % save_every == 0:
                log_confusion_matrix(sim_matrix, global_step, images.shape[0])

                raw_encoder   = trainer.encoder.module   if isinstance(trainer.encoder,   DDP) else trainer.encoder
                raw_projector = trainer.projector.module if isinstance(trainer.projector, DDP) else trainer.projector
                torch.save(
                    {
                        "encoder_state_dict":   raw_encoder.state_dict(),
                        "projector_state_dict": raw_projector.state_dict(),
                        "encoder_ema_state_dict": trainer.encoder_ema.state_dict(),
                        "projector_ema_state_dict": trainer.projector_ema.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "scheduler_state_dict": trainer.scheduler.state_dict(),
                        "scaler_state_dict":    trainer.scaler.state_dict(),
                        "global_step":          global_step,
                    },
                    os.path.join(config["training"]["output_dir"], f"epg_stage1_step_{global_step}.pth"),
                )

    if rank == 0:
        pbar.close()

    cleanup_ddp()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)