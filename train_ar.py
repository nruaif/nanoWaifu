import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import argparse
from tqdm.auto import tqdm
import wandb
from torchvision.utils import make_grid
import math
import time

from vae import CategoricalVAE
from ar_transformer import ARTransformer, HAS_FLEX
if HAS_FLEX:
    from torch.nn.attention.flex_attention import create_block_mask
from dataset import WDSLoader

# --- Utilities ---

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, local_rank, world_size, device
    return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, rank, output_dir, step, config):
    if rank != 0: return
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f'ar_ckpt_step_{step}.pth')
    torch.save({
        "model_state_dict": (model.module if hasattr(model, 'module') else model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step,
        "config": config,
    }, ckpt_path)
    print(f"✅ Checkpoint saved: {ckpt_path}")

def get_class_indices(prompts, class_map, device):
    all_indices, offsets = [], [0]
    for p in prompts:
        indices = [class_map.get(t, 0) for t in p.split()] or [0]
        all_indices.extend(indices)
        offsets.append(offsets[-1] + len(indices))
    return torch.tensor(all_indices, device=device), torch.tensor(offsets[:-1], device=device)

def get_2d_positions(lengths, resolutions, device):
    all_pos = []
    for (H, W) in resolutions:
        pos = [[0.0, 0.0], [0.0, 0.0]] # Cond, SOS
        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        xs, ys = torch.linspace(-xlim, xlim, W), torch.linspace(-ylim, ylim, H)
        for r in range(H):
            for c in range(W):
                pos.append([xs[c].item(), ys[r].item()])
        all_pos.extend(pos)
    return torch.tensor(all_pos, device=device).unsqueeze(0)

# --- Core Logic ---

def prepare_batch(batch, vae, device):
    packed_tokens, packed_prompts, resolutions = [], [], []
    for img, prompt, _ in batch:
        img = img.unsqueeze(0).to(device)
        tokens = vae.encode_to_indices(img) # (1, H, W, 256)
        H, W = tokens.size(1), tokens.size(2)
        packed_tokens.append(tokens.view(-1, 256))
        packed_prompts.append(prompt)
        resolutions.append((H, W))
    
    tokens_flat = torch.cat(packed_tokens, dim=0).unsqueeze(0)
    lengths = [len(t) + 2 for t in packed_tokens]
    return tokens_flat, packed_prompts, lengths, resolutions

def calculate_loss(pred_x, target_latents):
    # pred_x: (B, L, 256)
    # target_latents: (B, L, 256)
    return F.mse_loss(pred_x, target_latents)

@torch.no_grad()
def sample_and_log(model, vae, class_map, prompts, device, config, step):
    model.eval()
    class_indices, offsets = get_class_indices(prompts[:8], class_map, device)
    
    # Grid size (spatial)
    grid_size = config['training']['image_size'] // 32
    # Latent channels (discrete)
    latent_discrete = config['model']['latent_discrete']
    
    model_inner = model.module if hasattr(model, 'module') else model
    patch_latents = model_inner.generate(
        class_indices, max_patches=grid_size**2, device=device
    )
    
    patch_latents = patch_latents.view(-1, grid_size, grid_size, latent_discrete)
    # Decode from latents {-1, 0, 1}
    images = (vae.decode_from_latents(patch_latents) / 2 + 0.5).clamp(0, 1)
    grid = make_grid(images, nrow=4)
    wandb.log({"samples": wandb.Image(grid, caption=f"Step {step}")}, step=step)
    model.train()

# --- Main Training ---

def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    with open(config_path, 'r') as f: config = yaml.safe_load(f)

    if rank == 0: wandb.init(project=config.get('wandb_project', 'nanoWaifu-AR'), config=config)

    # Models
    vae = CategoricalVAE(
        latent_discrete=config['model']['latent_discrete'],
        latent_continuous=config['model']['latent_continuous']
    ).to(device).eval()
    if 'vae_path' in config['model']: vae.load_pretrained(config['model']['vae_path'], device=device)
    vae.requires_grad_(False)
    
    # Add helper to decode from latents directly if not present
    if not hasattr(vae, 'decode_from_latents'):
        def decode_from_latents(latents, z_continuous=None):
            # latents: (B, H, W, C)
            z_discrete = latents.permute(0, 3, 1, 2).contiguous()
            if z_continuous is None:
                z_continuous = torch.zeros(
                    (latents.shape[0], vae.latent_continuous, latents.shape[1], latents.shape[2]),
                    device=latents.device, dtype=z_discrete.dtype
                )
            return vae.vae.decode(z_discrete, z_continuous)
        vae.decode_from_latents = decode_from_latents

    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'],
        csv_path=config['data'].get('csv_path'),
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    dataloader = wds_loader.make_loader()
    class_map = wds_loader.class_map or {"unknown": 0}

    model = ARTransformer(
        num_classes=len(class_map) + 1,
        latent_dim=config['model']['latent_discrete'],
        dim=config['model'].get('dim', 512),
        depth=config['model'].get('depth', 12),
        num_heads=config['model'].get('heads', 8),
        max_seq_len=4096,
        dropout=config['training'].get('dropout', 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    if is_ddp: model = DDP(model, device_ids=[local_rank])

    # Loop
    global_step, max_steps = 0, config['training'].get('max_train_steps', 1000000)
    data_iter = iter(dataloader)
    if rank == 0: pbar = tqdm(range(max_steps), desc="Training")

    while global_step < max_steps:
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(dataloader); batch = next(data_iter)

        # Forward
        tokens_flat, prompts, lengths, resolutions = prepare_batch(batch, vae, device)
        # Convert indices {0, 1, 2} to latents {-1, 0, 1}
        latents_flat = tokens_flat.float() - 1.0
        
        class_indices, offsets = get_class_indices(prompts, class_map, device)
        positions = get_2d_positions(lengths, resolutions, device)
        
        # In ARTransformer, forward(patch_latents, ...) returns pred_x for all positions
        # x[1] (SOS) predicts latents[0], x[2] (P0) predicts latents[1], etc.
        pred_x = model(latents_flat, class_indices, positions, offsets=offsets)
        
        # Shift to align: pred_x[:, 1:] predicts latents_flat[:, :]
        loss = calculate_loss(pred_x[:, 1:], latents_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        if rank == 0:
            pbar.update(1)
            if global_step % config['training']['log_every_steps'] == 0:
                wandb.log({"loss": loss.item(), "grad_norm": grad_norm.item()}, step=global_step)
            if global_step % config['training']['save_image_every_steps'] == 0:
                save_checkpoint(model, optimizer, rank, config['training']['output_dir'], global_step, config)
                sample_and_log(model, vae, class_map, prompts, device, config, global_step)

    if is_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_ar.yaml")
    train(parser.parse_args().config)
