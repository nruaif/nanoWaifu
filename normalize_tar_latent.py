"""
Normalize cached latent tars using VAE stats (vae_stats.pt).

If your cached latents were created without --stats_path (or with --stats_path missing),
they are **unnormalized** raw VAE outputs (128ch, after pixel_unshuffle for standard VAE).
This script converts them to normalized form:  (latent - mean) / std

Usage:
    python normalize_tar_latent.py --input "cached_latents/latents-*.tar" \
                                   --output_dir cached_latents_norm \
                                   --stats_path vae_stats.pt

    # Single file:
    python normalize_tar_latent.py --input "C:/Downloads/00001.tar" --output_dir ./fixed --stats_path vae_stats.pt

The output is a new set of WebDataset .tar shards with the same structure:
  - "latent.npy": float16 normalized latent [128, H', W']
  - "prompt.txt": unchanged prompt

If latents are already normalized (mean ~0, std ~1), the script will still re-apply the formula
unless you pass --skip-if-normalized.
"""
import argparse
import glob
import os
import io
import uuid
import tarfile
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm


def load_stats(stats_path, device="cpu"):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"vae_stats.pt not found at {stats_path}. "
                                f"Generate it via cache_latents.py or train.py first run.")
    stats = torch.load(stats_path, map_location=device)
    mean = stats["mean"]
    std = stats["std"]
    # Ensure shape [1, C, 1, 1] for broadcasting
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)
    if mean.ndim == 3:
        mean = mean.unsqueeze(0)
    if std.ndim == 3:
        std = std.unsqueeze(0)
    print(f"Loaded stats from {stats_path}: mean shape {tuple(mean.shape)}, std shape {tuple(std.shape)}")
    print(f"  mean sample {mean.flatten()[:5].tolist()}, std sample {std.flatten()[:5].tolist()}")
    return mean.float(), std.float()


def normalize_latent(latent_np, mean, std, eps=1e-6):
    """
    latent_np: np.ndarray [C, H, W] float16/float32
    mean, std: torch.Tensor [1, C, 1, 1] or [C]
    Returns normalized np.ndarray same shape, float16
    """
    # Convert to torch for broadcasting, keep float32 for precision
    t = torch.from_numpy(latent_np).float()  # [C, H, W]
    # mean/std to same dtype/device as t, squeeze batch dim
    if mean.shape[0] == 1:
        mean = mean.squeeze(0)  # [C,1,1]
        std = std.squeeze(0)
    # Ensure std not zero
    std = torch.clamp(std, min=eps)
    # t: [C,H,W], mean/std: [C,1,1]
    t = (t - mean) / std
    return t.numpy().astype(np.float16)


def is_already_normalized(latent_np, thresh_mean=0.5, thresh_std=0.3):
    """
    Heuristic: check if latent is already normalized (mean ~0, std ~1)
    Returns True if likely normalized.
    """
    m = float(latent_np.mean())
    s = float(latent_np.std())
    return abs(m) < thresh_mean and abs(s - 1.0) < thresh_std


def process_tar(input_path, output_dir, mean, std, writer, skip_if_normalized=False):
    """
    Read one input .tar, normalize each latent.npy, write to ShardWriter.
    """
    # Count samples
    with tarfile.open(input_path, "r") as tf:
        members = tf.getmembers()
        # Each sample has latent.npy + prompt.txt, grouped by __key__
        # We'll use wds to properly group by keys
        pass

    # Use webdataset to iterate properly grouped samples
    dataset = wds.WebDataset(input_path, handler=wds.warn_and_continue)
    count = 0
    pbar = tqdm(desc=f"Normalizing {os.path.basename(input_path)}", unit="sample")
    for sample in dataset:
        if "latent.npy" not in sample:
            # Skip non-latent samples (e.g., raw webp+json)
            print(f"  Warning: sample {sample.get('__key__','?')} has no latent.npy, skipping")
            continue
        latent_data = sample["latent.npy"]
        if isinstance(latent_data, bytes):
            latent_np = np.load(io.BytesIO(latent_data))
        elif isinstance(latent_data, np.ndarray):
            latent_np = latent_data
        else:
            # torch tensor?
            latent_np = np.array(latent_data)

        if skip_if_normalized and is_already_normalized(latent_np):
            normalized = latent_np.astype(np.float16)
        else:
            normalized = normalize_latent(latent_np, mean, std)

        prompt = sample.get("prompt.txt", sample.get("prompt", ""))
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if not isinstance(prompt, str):
            prompt = str(prompt)

        writer.write({
            "__key__": sample.get("__key__", uuid.uuid4().hex),
            "latent.npy": normalized,
            "prompt.txt": prompt,
        })
        count += 1
        pbar.update(1)
    pbar.close()
    return count


def main():
    parser = argparse.ArgumentParser(description="Normalize unnormalized cached latent tars via vae_stats.pt")
    parser.add_argument("--input", type=str, required=True,
                        help="Input tar glob, e.g. 'cached_latents/latents-*.tar' or single file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for normalized tars")
    parser.add_argument("--stats_path", type=str, default="vae_stats.pt",
                        help="Path to vae_stats.pt (from train.py or cache_latents.py)")
    parser.add_argument("--maxcount", type=int, default=100000,
                        help="Max samples per output shard (ShardWriter)")
    parser.add_argument("--maxsize", type=int, default=10**9,
                        help="Max bytes per output shard (~1GB)")
    parser.add_argument("--skip-if-normalized", action="store_true",
                        help="Skip re-normalizing if latent already looks normalized")
    args = parser.parse_args()

    # Expand glob
    input_files = glob.glob(args.input)
    if not input_files:
        # Try as direct path
        if os.path.exists(args.input):
            input_files = [args.input]
        else:
            raise FileNotFoundError(f"No files matched input pattern: {args.input}")
    input_files = sorted(input_files)
    print(f"Found {len(input_files)} input tar(s):")
    for f in input_files[:5]:
        print(f"  {f}")
    if len(input_files) > 5:
        print(f"  ... and {len(input_files)-5} more")

    mean, std = load_stats(args.stats_path)

    os.makedirs(args.output_dir, exist_ok=True)
    pattern = os.path.join(args.output_dir, "latents-%05d.tar")
    # Start shard 0, or resume if exists
    existing = glob.glob(os.path.join(args.output_dir, "latents-*.tar"))
    start_shard = 0
    if existing:
        idxs = []
        for f in existing:
            try:
                idxs.append(int(os.path.basename(f).split("-")[1].split(".")[0]))
            except:
                pass
        if idxs:
            start_shard = max(idxs) + 1
            print(f"Resuming at shard {start_shard} (found {len(idxs)} existing shards)")

    writer = wds.ShardWriter(pattern, maxcount=args.maxcount, maxsize=args.maxsize, start_shard=start_shard)

    total = 0
    for inp in input_files:
        cnt = process_tar(inp, args.output_dir, mean, std, writer, skip_if_normalized=args.skip_if_normalized)
        print(f"  -> {cnt} samples normalized from {inp}")
        total += cnt

    writer.close()
    print(f"\nDone! {total} samples written to {args.output_dir}")
    if total == 0:
        print("Warning: 0 samples processed. Check that input tars contain latent.npy (not raw webp+json). "
              "Raw tars need to be cached via cache_latents.py first.")


if __name__ == "__main__":
    main()
