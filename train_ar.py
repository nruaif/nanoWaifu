import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import argparse
import glob  # Added for auto-resume
import re  # Added for auto-resume
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


def save_checkpoint(model, optimizer, rank, output_dir, step, config, max_keep=3):
    """Saves checkpoint and deletes older ones to save disk space."""
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

    # Prune old checkpoints
    checkpoints = glob.glob(os.path.join(output_dir, 'ar_ckpt_step_*.pth'))
    # Sort numerically by step number
    checkpoints.sort(key=lambda x: int(re.search(r'_step_(\d+)\.pth', x).group(1)))

    while len(checkpoints) > max_keep:
        oldest_ckpt = checkpoints.pop(0)
        os.remove(oldest_ckpt)
        print(f"🗑️ Deleted old checkpoint: {oldest_ckpt}")


def get_class_indices(prompts, class_map, device):
    all_indices, offsets = [], [0]
    for p in prompts:
        indices = [class_map.get(t, 0) for t in p.split()] or [0]
        all_indices.extend(indices)
        offsets.append(offsets[-1] + len(indices))
    return torch.tensor(all_indices, device=device), torch.tensor(offsets[:-1], device=device)


def get_2d_positions(resolutions, max_len, device):
    all_pos = []
    for (H, W) in resolutions:
        pos = [[0.0, 0.0], [0.0, 0.0]]  # Cond, SOS
        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        xs, ys = torch.linspace(-xlim, xlim, W), torch.linspace(-ylim, ylim, H)
        for r in range(H):
            for c in range(W):
                pos.append([xs[c].item(), ys[r].item()])
        pad_len = max_len - (H * W)
        for _ in range(pad_len):
            pos.append([0.0, 0.0])
        all_pos.append(pos)
    return torch.tensor(all_pos, device=device)


# --- Core Logic ---

def prepare_batch(batch, vae, device):
    packed_tokens, packed_prompts, resolutions = [], [], []
    for img, prompt, _ in batch:
        img = img.unsqueeze(0).to(device)
        tokens = vae.encode_to_indices(img)  # (1, H, W, 256)
        H, W = tokens.size(1), tokens.size(2)
        packed_tokens.append(tokens.view(-1, 256))
        packed_prompts.append(prompt)
        resolutions.append((H, W))

    max_len = max([t.size(0) for t in packed_tokens])
    padded_tokens = []
    masks = []

    for t in packed_tokens:
        pad_len = max_len - t.size(0)
        padded_t = F.pad(t, (0, 0, 0, pad_len))
        mask = torch.cat([torch.ones(t.size(0)), torch.zeros(pad_len)])
        padded_tokens.append(padded_t)
        masks.append(mask)

    tokens_batched = torch.stack(padded_tokens).to(device)
    masks_batched = torch.stack(masks).to(device).bool()
    return tokens_batched, masks_batched, packed_prompts, resolutions


def calculate_loss(pred_x, target_latents):
    return F.mse_loss(pred_x, target_latents)


@torch.no_grad()
def sample_and_log(model, vae, class_map, prompts, device, config, step):
    model.eval()

    # Restrict to 4 prompts to create a 2x2 grid
    sample_prompts = prompts[:4]
    if len(sample_prompts) < 4:
        sample_prompts = (sample_prompts * 4)[:4]  # Pad if batch is too small

    class_indices, offsets = get_class_indices(sample_prompts, class_map, device)

    # Grid size (spatial)
    grid_size = config['training']['image_size'] // 32
    # Latent channels (discrete)
    latent_discrete = config['model']['latent_discrete']

    model_inner = model.module if hasattr(model, 'module') else model

    patch_latents = model_inner.generate(
        class_indices,
        grid_H=grid_size,
        grid_W=grid_size,
        device=device,
    )

    patch_latents = patch_latents.view(-1, grid_size, grid_size, latent_discrete)
    # Decode from latents {-1, 0, 1}
    images = (vae.decode_from_latents(patch_latents) / 2 + 0.5).clamp(0, 1)

    # Make a 2x2 grid (nrow=2)
    grid = make_grid(images, nrow=2)

    if wandb.run is not None:
        wandb.log({"samples": wandb.Image(grid, caption=f"Step {step}")}, step=step)

    model.train()


# --- Main Training ---

def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01,
                                  betas=(0.9, 0.95))

    # --- Auto-Resume Logic ---
    global_step = 0
    output_dir = config['training'].get('output_dir', './checkpoints')

    if os.path.exists(output_dir):
        checkpoints = glob.glob(os.path.join(output_dir, 'ar_ckpt_step_*.pth'))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r'_step_(\d+)\.pth', x).group(1)))
            print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint['global_step'] - 1
            if rank == 0:
                print(f"✅ Successfully loaded weights and resumed at step {global_step}")

    if is_ddp: model = DDP(model, device_ids=[local_rank])

    # Loop
    max_steps = config['training'].get('max_train_steps', 1000000)
    data_iter = iter(dataloader)
    if rank == 0:
        pbar = tqdm(total=max_steps, initial=global_step, desc="Training")
    #model.compile()
    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader); batch = next(data_iter)

        # Forward
        tokens_batched, masks_batched, prompts, resolutions = prepare_batch(batch, vae, device)
        latents_batched = tokens_batched.float() - 1.0

        class_indices, offsets = get_class_indices(prompts, class_map, device)
        max_len = latents_batched.size(1)
        positions = get_2d_positions(resolutions, max_len, device)

        B, L, _ = latents_batched.shape
        t = torch.rand(B, L, 1, device=device)
        noise = torch.randn_like(latents_batched)

        x_t = (1 - t) * noise + t * latents_batched

        padding_mask = torch.cat([torch.ones(B, 2, device=device, dtype=torch.bool), masks_batched[:, :-1]], dim=1)
        L_x = padding_mask.size(1)
        valid_lens = padding_mask.sum(dim=1)

        def causal_padding_mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            not_padding = (q_idx < valid_lens[b]) & (kv_idx < valid_lens[b])
            return causal & not_padding

        causal_mask = torch.triu(torch.ones(L_x, L_x, device=device), diagonal=1).bool()
        invalid = (~padding_mask.unsqueeze(2)) | (~padding_mask.unsqueeze(1)) | causal_mask.unsqueeze(0)
        block_mask = torch.zeros(B, 1, L_x, L_x, device=device)
        block_mask.masked_fill_(invalid.unsqueeze(1), -float('inf'))
        block_mask = create_block_mask(
            causal_padding_mask_mod,
            B=B, H=None, Q_LEN=L_x, KV_LEN=L_x, device=device, _compile=True
        )
        grid_HW = torch.tensor(resolutions, device=device)

        pred_x = model(
            latents_batched, class_indices, positions,
            grid_HW=grid_HW,  # <-- pass it in
            offsets=offsets,
            block_mask=block_mask,
            x_t=x_t, t=t
        )

        loss = F.mse_loss(pred_x[masks_batched], latents_batched[masks_batched])

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
                # Set max_keep here (defaulting to 3, you can change it)
                save_checkpoint(model, optimizer, rank, output_dir, global_step, config, max_keep=3)
                sample_and_log(model, vae, class_map, prompts, device, config, global_step)

    if is_ddp: dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_ar.yaml")
    train(parser.parse_args().config)