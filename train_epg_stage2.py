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
import random
from dataset import WDSLoader
from model_epg import EPGEncoder, EPGDecoder, EPGModel

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

def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    if rank != 0: builtins.print = lambda *args, **kwargs: None

    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    os.makedirs(config['training']['output_dir'], exist_ok=True)


    dataloader = WDSLoader(url=config['data']['webdataset_url'], csv_path=config['data'].get('csv_path'), image_size=config['training']['image_size'], batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers']).make_loader()
    from model_epg import TagProcessor
    # Models
    embed_dim = config['model'].get('embed_dim', 768)
    depth = config['model'].get('depth', 12)
    num_heads = config['model'].get('num_heads', 12)
    num_classes = config['model'].get('num_classes', 12476)
    
    encoder = EPGEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, num_classes=num_classes).to(device)
    decoder = EPGDecoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads).to(device)
    model = EPGModel(encoder, decoder).to(device)
    
    tag_processor = TagProcessor("tags.txt")

    # Load pre-trained encoder
    stage1_ckpt = config.get('stage1_ckpt')
    if stage1_ckpt and os.path.exists(stage1_ckpt):
        print(f"Loading Stage 1 pre-trained encoder from {stage1_ckpt}")
        ckpt = torch.load(stage1_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training'].get('weight_decay', 0.01))

    global_step, max_steps = 0, config['training'].get('max_train_steps', 1000000)
    if rank == 0:
        wandb.init(project="EPG-Stage2-PFM-x0", config=config)
        pbar = tqdm(range(max_steps), desc="EPG Stage 2")

    data_iter = iter(dataloader)
    while global_step < max_steps:
        model.train()
        optimizer.zero_grad()
        try: batch = next(data_iter)
        except StopIteration: batch = next(iter(dataloader))
            
        images, prompts, _ = batch
        images = images.to(device)
        y_indices, y_offsets = tag_processor.process_prompts(prompts, device)
        
        # Diffusion / Flow Matching Training (x0 prediction)
        t = torch.rand((images.shape[0],), device=device)
        noise = torch.randn_like(images)
        t_scaled = 1000 * 0.25 * torch.log(t.clamp(min=1e-8))
        
        sigma_data = 0.5
        precond = 1.0 / torch.sqrt(t**2 + sigma_data**2)
        xt = ((1 - t.view(-1, 1, 1, 1)) * images + t.view(-1, 1, 1, 1) * noise) * precond.view(-1, 1, 1, 1)
        
        pred_x0 = model(xt, t_scaled, y_indices=y_indices, y_offsets=y_offsets, stage=2)
        loss = F.mse_loss(pred_x0, images)
        
        loss.backward()
        
        if config['training'].get('grad_clip', 0.0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            
        optimizer.step()
        
        global_step += 1
        if rank == 0:
            pbar.update(1)
            if global_step % config['training'].get('log_every_steps', 100) == 0:
                wandb.log({"loss": loss.item()}, step=global_step)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if global_step % config['training'].get('save_image_every_steps', 5000) == 0:
                model_to_sample = model.module if hasattr(model, 'module') else model
                samples = model_to_sample.sample_flow(image_size=config['training']['image_size'], batch_size=16, device=device, y_indices=y_indices[:16], y_offsets=y_offsets[:16])
                wandb.log({"samples": wandb.Image(make_grid(samples, nrow=4))}, step=global_step)
                torch.save(model.state_dict(), os.path.join(config['training']['output_dir'], f"epg_stage2_step_{global_step}.pth"))

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    train(parser.parse_args().config)
