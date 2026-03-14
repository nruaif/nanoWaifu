import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import argparse
import glob
import re
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


def save_checkpoint(model, optimizer, rank, output_dir, step, config, max_keep=3):
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f'ar_ckpt_step_{step}.pth')
    torch.save({
        "model_state_dict": (model.module if hasattr(model, 'module') else model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step,
        "config": config,
    }, ckpt_path)
    print(f"✅ Checkpoint saved: {ckpt_path}")

    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, 'ar_ckpt_step_*.pth')),
        key=lambda x: int(re.search(r'_step_(\d+)\.pth', x).group(1))
    )
    while len(checkpoints) > max_keep:
        oldest = checkpoints.pop(0)
        os.remove(oldest)
        print(f"🗑️  Deleted old checkpoint: {oldest}")


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------

def get_class_indices(prompts, class_map, device):
    all_indices, offsets = [], [0]
    for p in prompts:
        indices = [class_map.get(t, 0) for t in p.split()] or [0]
        all_indices.extend(indices)
        offsets.append(offsets[-1] + len(indices))
    return (
        torch.tensor(all_indices, device=device),
        torch.tensor(offsets[:-1], device=device),
    )


def get_2d_positions(resolutions, max_patch_len, device):
    """
    Build position tensors for the full sequence including special tokens.

    Sequence layout per sample:
        [Cond, Size, SOS, P1, P2, … PL, EOS, <pad>…]
         idx0  idx1  idx2  idx3 …

    Special tokens (Cond, Size, SOS, EOS) all get position [0, 0].
    Patch positions use aspect-ratio-aware linspace matching training.

    Args:
        resolutions:    list of (H, W) patch-grid sizes, length B
        max_patch_len:  maximum number of patches across the batch (= L)
        device:

    Returns:
        positions: (B, L+4, 2)   — L patches + 3 prefix tokens + 1 EOS
    """
    all_pos = []
    for (H, W) in resolutions:
        pos = [
            [0.0, 0.0],   # Cond  (idx 0)
            [0.0, 0.0],   # Size  (idx 1)
            [0.0, 0.0],   # SOS   (idx 2)
        ]
        xlim = math.sqrt(W / H)
        ylim = math.sqrt(H / W)
        xs = torch.linspace(-xlim, xlim, W)
        ys = torch.linspace(-ylim, ylim, H)
        for r in range(H):
            for c in range(W):
                pos.append([xs[c].item(), ys[r].item()])

        # Padding slots (zero position, will be masked out)
        pad_len = max_patch_len - (H * W)
        for _ in range(pad_len):
            pos.append([0.0, 0.0])

        pos.append([0.0, 0.0])   # EOS  (last)
        all_pos.append(pos)

    return torch.tensor(all_pos, dtype=torch.float32, device=device)
    # shape: (B, 3 + max_patch_len + 1, 2)  =  (B, L+4, 2)


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def prepare_batch(batch, vae, device):
    """
    Encode images to discrete latents, pad to uniform length within the batch.

    Returns:
        tokens_batched:  (B, L, 256)  float latents padded to max_len
        masks_batched:   (B, L)       bool, True = real patch
        prompts:         list[str]
        resolutions:     list[(H, W)] patch-grid sizes
        grid_HW:         (B, 2)       int tensor of (H, W) per sample
    """
    packed_tokens, packed_prompts, resolutions = [], [], []

    for img, prompt, _ in batch:
        img = img.unsqueeze(0).to(device)
        tokens = vae.encode_to_indices(img)   # (1, H, W, 256)
        H, W = tokens.size(1), tokens.size(2)
        packed_tokens.append(tokens.view(-1, 256))
        packed_prompts.append(prompt)
        resolutions.append((H, W))

    max_len = max(t.size(0) for t in packed_tokens)
    padded_tokens, masks = [], []

    for t in packed_tokens:
        pad_len = max_len - t.size(0)
        padded_tokens.append(F.pad(t, (0, 0, 0, pad_len)))
        masks.append(torch.cat([torch.ones(t.size(0)), torch.zeros(pad_len)]))

    tokens_batched = torch.stack(padded_tokens).to(device)           # (B, L, 256)
    masks_batched  = torch.stack(masks).to(device).bool()            # (B, L)
    grid_HW        = torch.tensor(resolutions, device=device)        # (B, 2)

    return tokens_batched, masks_batched, packed_prompts, resolutions, grid_HW


# ---------------------------------------------------------------------------
# Attention mask
# ---------------------------------------------------------------------------

def build_block_mask(masks_batched, device):
    """
    Build a causal + padding block mask for FlexAttention.

    Sequence layout: [Cond, Size, SOS, P1 … PL-1, EOS]  length = L+3
    (we feed PL-1 not PL into the transformer input, EOS is appended at end)

    The 3 prefix tokens (Cond, Size, SOS) are always valid.
    The 1 suffix token (EOS) is always valid.
    Patch slots follow masks_batched[:, :-1] (shifted by one for AR input).
    """
    B, L = masks_batched.shape

    # Sequence length inside the transformer = L + N_PREFIX + N_SUFFIX - 1 patch
    # Input is: [Cond, Size, SOS, P1..PL-1, EOS]  -> L - 1 patches + 3 + 1 = L + 3
    prefix_mask = torch.ones(B, 3, device=device, dtype=torch.bool)   # Cond, Size, SOS
    patch_mask  = masks_batched[:, :-1]                                # P1..PL-1
    eos_mask    = torch.ones(B, 1, device=device, dtype=torch.bool)    # EOS
    padding_mask = torch.cat([prefix_mask, patch_mask, eos_mask], dim=1)  # (B, L+3)

    L_x = padding_mask.size(1)
    valid_lens = padding_mask.sum(dim=1)   # (B,)  number of valid tokens per sample

    def causal_padding_mask_mod(b, h, q_idx, kv_idx):
        causal      = q_idx >= kv_idx
        not_padding = (q_idx < valid_lens[b]) & (kv_idx < valid_lens[b])
        return causal & not_padding

    block_mask = create_block_mask(
        causal_padding_mask_mod,
        B=B, H=None, Q_LEN=L_x, KV_LEN=L_x,
        device=device, _compile=True,
    )
    return block_mask


# ---------------------------------------------------------------------------
# Sampling / logging
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_and_log(model, vae, class_map, prompts, device, config, step):
    model.eval()

    sample_prompts = (prompts * 4)[:4]
    class_indices, offsets = get_class_indices(sample_prompts, class_map, device)

    grid_size     = config['training']['image_size'] // 32
    latent_dim    = config['model']['latent_discrete']

    model_inner = model.module if hasattr(model, 'module') else model
    patch_latents = model_inner.generate(
        class_indices,
        grid_H=grid_size,
        grid_W=grid_size,
        device=device,
    )   # (B, H*W, latent_dim)

    B = patch_latents.size(0)
    patch_latents = patch_latents.view(B, grid_size, grid_size, latent_dim)
    images = (vae.decode_from_latents(patch_latents) / 2 + 0.5).clamp(0, 1)

    grid = make_grid(images, nrow=2)
    if wandb.run is not None:
        wandb.log({"samples": wandb.Image(grid, caption=f"Step {step}")}, step=step)

    model.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-AR'), config=config)

    # --- VAE (frozen) ---
    vae = CategoricalVAE(
        latent_discrete=config['model']['latent_discrete'],
        latent_continuous=config['model']['latent_continuous'],
    ).to(device).eval()
    if 'vae_path' in config['model']:
        vae.load_pretrained(config['model']['vae_path'], device=device)
    vae.requires_grad_(False)

    if not hasattr(vae, 'decode_from_latents'):
        def decode_from_latents(latents, z_continuous=None):
            z_discrete = latents.permute(0, 3, 1, 2).contiguous()
            if z_continuous is None:
                z_continuous = torch.zeros(
                    (latents.shape[0], vae.latent_continuous,
                     latents.shape[1], latents.shape[2]),
                    device=latents.device, dtype=z_discrete.dtype,
                )
            return vae.vae.decode(z_discrete, z_continuous)
        vae.decode_from_latents = decode_from_latents

    # --- Dataset ---
    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'],
        csv_path=config['data'].get('csv_path'),
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
    )
    dataloader = wds_loader.make_loader()
    class_map  = wds_loader.class_map or {"unknown": 0}

    # --- Model ---
    model = ARTransformer(
        num_classes=len(class_map) + 1,
        latent_dim=config['model']['latent_discrete'],
        dim=config['model'].get('dim', 512),
        depth=config['model'].get('depth', 12),
        num_heads=config['model'].get('heads', 8),
        max_seq_len=4096,
        dropout=config['training'].get('dropout', 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # --- Auto-resume ---
    global_step = 0
    output_dir  = config['training'].get('output_dir', './checkpoints')

    if os.path.exists(output_dir):
        checkpoints = glob.glob(os.path.join(output_dir, 'ar_ckpt_step_*.pth'))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(re.search(r'_step_(\d+)\.pth', x).group(1)))
            print(f"🔄 Resuming from: {latest}")
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            global_step = ckpt['global_step']
            if rank == 0:
                print(f"✅ Resumed at step {global_step}")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # --- Training loop ---
    max_steps = config['training'].get('max_train_steps', 1_000_000)
    data_iter = iter(dataloader)

    if rank == 0:
        pbar = tqdm(total=max_steps, initial=global_step, desc="Training")

    while global_step < max_steps:
        # --- Data ---
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        tokens_batched, masks_batched, prompts, resolutions, grid_HW = \
            prepare_batch(batch, vae, device)

        # Convert discrete tokens {0,1,2} -> continuous {-1,0,1}
        latents_batched = tokens_batched.float() - 1.0   # (B, L, 256)
        B, L, _ = latents_batched.shape

        # --- Conditioning ---
        class_indices, offsets = get_class_indices(prompts, class_map, device)

        # Positions: (B, L+4, 2)  — 3 prefix + L patches + 1 EOS
        positions = get_2d_positions(resolutions, L, device)

        # --- Flow matching noise ---
        # t and x_t have shape (B, L, *) matching patch count only.
        # The model internally handles the extra special tokens.
        t     = torch.rand(B, L, 1, device=device)
        noise = torch.randn_like(latents_batched)
        x_t   = (1 - t) * noise + t * latents_batched

        # --- Attention mask ---
        block_mask = build_block_mask(masks_batched, device)

        # --- Forward ---
        # pred_x: (B, L, latent_dim)  — already aligned with latents_batched
        pred_x = model(
            latents_batched,
            class_indices,
            positions,
            grid_HW=grid_HW,
            offsets=offsets,
            block_mask=block_mask,
            x_t=x_t,
            t=t,
        )

        # Loss only over real (non-padded) patches
        loss = F.mse_loss(pred_x[masks_batched], latents_batched[masks_batched])

        # --- Backward ---
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
                save_checkpoint(model, optimizer, rank, output_dir, global_step, config, max_keep=3)
                sample_and_log(model, vae, class_map, prompts, device, config, global_step)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_ar.yaml")
    train(parser.parse_args().config)