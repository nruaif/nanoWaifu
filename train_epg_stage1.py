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
from tqdm.auto import tqdm
import wandb
import builtins
import copy
from dataset import WDSLoader
from model_epg import EPGEncoder, EPGProjector, EPGModel, TagProcessor
from torchvision import transforms
from siglip import SupConLoss
import matplotlib.pyplot as plt
import random

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

def update_ema(target_model, source_model, beta):
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(beta * target_param.data + (1 - beta) * source_param.data)

def log_confusion_matrix(sim_matrix, step, batch_size):
    """Logs Positive vs Negative similarity between view groups to detect collapse."""
    k = sim_matrix.shape[0] // batch_size
    reshaped = sim_matrix.view(k, batch_size, k, batch_size)
    
    pos_sim = np.zeros((k, k))
    neg_sim = np.zeros((k, k))
    
    for i in range(k):
        for j in range(k):
            block = reshaped[i, :, j, :] # (B, B)
            pos_sim[i, j] = block.diagonal().mean().item()
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=block.device)
            neg_sim[i, j] = block[mask].mean().item()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pos_sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    views = ["Aug", "Noise", "Label"]
    ax.set_xticks(range(k)); ax.set_xticklabels(views)
    ax.set_yticks(range(k)); ax.set_yticklabels(views)
    
    for i in range(k):
        for j in range(k):
            color = "w" if pos_sim[i, j] < 0.5 else "k"
            ax.text(j, i, f"Pos: {pos_sim[i, j]:.2f}\nNeg: {neg_sim[i, j]:.2f}", 
                    ha="center", va="center", color=color, fontsize=9, fontweight='bold')
            
    ax.set_title(f"Group Alignment: Positive vs Negative (Step {step})")
    wandb.log({"val/alignment_contrast": wandb.Image(fig)}, step=step)
    plt.close(fig)

class EPGStage1Trainer:
    def __init__(self, config, device, rank):
        self.config = config
        self.device = device
        self.rank = rank
        
        self.embed_dim = config['model'].get('embed_dim', 768)
        self.proj_dim = config['model'].get('proj_dim', 256)
        self.ema_beta = config['training'].get('ema_beta', 0.99)
        self.num_classes = config['model'].get('num_classes', 12476)
        
        self.encoder = EPGEncoder(embed_dim=self.embed_dim, depth=config['model'].get('depth', 12), num_heads=config['model'].get('num_heads', 12), num_classes=self.num_classes).to(device)
        self.projector = EPGProjector(embed_dim=self.embed_dim, proj_dim=self.proj_dim).to(device)
        
        self.encoder_ema = copy.deepcopy(self.encoder).to(device)
        self.projector_ema = copy.deepcopy(self.projector).to(device)
        for m in [self.encoder_ema, self.projector_ema]:
            for param in m.parameters(): param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.projector.parameters()), lr=config['training']['learning_rate'], weight_decay=config['training'].get('weight_decay', 0.05))
        
        # Compile only the loss module
        self.supcon = SupConLoss().to(device)
        if config['training'].get('compile', False):
            self.supcon = torch.compile(self.supcon)
            
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(config['training']['image_size'], scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def train_step(self, x, y_indices, y_offsets, y_labels, step, max_steps):
        self.encoder.train(); self.projector.train(); self.optimizer.zero_grad()
        B = x.shape[0]
        y_feat_online = self.projector(self.encoder.get_y_feat(y_indices, y_offsets))
        with torch.no_grad(): y_feat_target = self.projector_ema(self.encoder_ema.get_y_feat(y_indices, y_offsets))

        y1, y2 = self.aug(x), self.aug(x)
        t0 = torch.zeros(B, device=self.device)
        
        N = 1280; n = torch.randint(1, N, (B,), device=self.device); tn, tn_1 = n.float() / N, (n - 1).float() / N
        tn_scaled = 1000 * 0.25 * torch.log(tn.clamp(min=1e-8))
        tn_1_scaled = 1000 * 0.25 * torch.log(tn_1.clamp(min=1e-8))

        epsilon = torch.randn_like(x); sigma_data = 0.5
        xtn = (x + tn.view(-1, 1, 1, 1) * epsilon) * (1.0 / torch.sqrt(tn**2 + sigma_data**2)).view(-1, 1, 1, 1)
        xtn_1 = (x + tn_1.view(-1, 1, 1, 1) * epsilon) * (1.0 / torch.sqrt(tn_1**2 + sigma_data**2)).view(-1, 1, 1, 1)

        feat_im1 = self.encoder(y1, t0, y_indices, y_offsets); q_im1 = self.projector(feat_im1[:, 0])
        feat_n = self.encoder(xtn, tn_scaled, y_indices, y_offsets); qn = self.projector(feat_n[:, 0])

        with torch.no_grad():
            feat_im2 = self.encoder_ema(y2, t0, y_indices, y_offsets); q_im2 = self.projector_ema(feat_im2[:, 0])
            feat_n_1 = self.encoder_ema(xtn_1, tn_1_scaled, y_indices, y_offsets); qn_1 = self.projector_ema(feat_n_1[:, 0])

        loss, _, sim_matrix = self.supcon([q_im1, qn, y_feat_online], [q_im2, qn_1, y_feat_target], labels=y_labels)
        loss.backward()
        
        if self.config['training'].get('grad_clip', 0.0) > 0:
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.projector.parameters()), 
                                    self.config['training']['grad_clip'])
            
        self.optimizer.step()
        update_ema(self.encoder_ema, self.encoder, self.ema_beta)
        update_ema(self.projector_ema, self.projector, self.ema_beta)
        return {"loss": loss.item()}, sim_matrix

def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    if rank != 0: builtins.print = lambda *args, **kwargs: None
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    dataloader = WDSLoader(url=config['data']['webdataset_url'], csv_path=config['data'].get('csv_path'), image_size=config['training']['image_size'], batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers']).make_loader()
    trainer = EPGStage1Trainer(config, device, rank)
    tag_processor = TagProcessor("tags.txt")
    max_steps, global_step = config['training'].get('max_train_steps', 600000), 0
    if rank == 0: wandb.init(project="EPG-Stage1-SupCon", config=config); pbar = tqdm(range(max_steps), desc="EPG Stage 1")
    data_iter = iter(dataloader)
    while global_step < max_steps:
        try: batch = next(data_iter)
        except StopIteration: batch = next(iter(dataloader))
        images, prompts, _ = batch
        y_indices, y_offsets, y_labels = tag_processor.process_prompts(prompts.to(device) if hasattr(prompts, 'to') else prompts, device)
        metrics, sim_matrix = trainer.train_step(images.to(device), y_indices, y_offsets, y_labels, global_step, max_steps)
        global_step += 1
        if rank == 0:
            pbar.update(1)
            if global_step % config['training'].get('log_every_steps', 100) == 0: wandb.log(metrics, step=global_step); pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
            if global_step % config['training'].get('save_ckpt_every_steps', 5000) == 0:
                log_confusion_matrix(sim_matrix, global_step, images.shape[0])
                torch.save({"encoder_state_dict": trainer.encoder.state_dict(), "projector_state_dict": trainer.projector.state_dict(), "global_step": global_step}, os.path.join(config['training']['output_dir'], f"epg_stage1_step_{global_step}.pth"))
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=str, default="config.yaml"); train(parser.parse_args().config)
