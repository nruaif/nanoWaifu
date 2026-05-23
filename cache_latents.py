"""
Encode images from WebDataset tars into cached latents, grouped into 1GB WebDataset .tar shards.

Each output file is a .tar shard containing samples with:
  - "latent.npy": float16 array of shape [128, H', W']
  - "prompt.txt": The prompt string

Files are named: latents-00000.tar, latents-00001.tar, etc.

Usage:
    python cache_latents.py --input "/path/to/shards/00{001..020}.tar" \
                            --output_dir ./cached_latents \
                            --image_size 256 \
                            --bucket_size 128
"""
import argparse
import io
import json
import math
import os
import time

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF


# ── Bucket calculation (matches WDSLoader) ──────────────────────────────────

def make_buckets(image_size):
    target_area = image_size ** 2
    aspect_ratios = [1.0, 0.75, 1.33, 0.56, 1.78]
    buckets = []
    for ar in aspect_ratios:
        h = int(math.sqrt(target_area / ar))
        w = int(h * ar)
        h = (h // 16) * 16
        w = (w // 16) * 16
        if h > 0 and w > 0:
            buckets.append((h, w))
    return buckets


def find_best_bucket(w, h, buckets):
    img_ar = w / h
    bucket_ars = [bw / bh for bh, bw in buckets]
    best_idx = min(range(len(bucket_ars)), key=lambda i: abs(bucket_ars[i] - img_ar))
    return buckets[best_idx]


# ── Image preprocessing (deterministic center crop) ─────────────────────────

def preprocess_image(image, target_h, target_w):
    """Resize-to-cover then center-crop to target bucket size."""
    w, h = image.size
    img_ar = w / h
    target_ar = target_w / target_h

    if img_ar > target_ar:
        new_h = target_h
        new_w = int(target_h * img_ar)
    else:
        new_w = target_w
        new_h = int(target_w / img_ar)

    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))

    tensor = TF.to_tensor(image)
    tensor = (tensor - 0.5) * 2.0
    return tensor


# ── Extract prompt from sample ──────────────────────────────────────────────

def extract_prompt(sample):
    """Pull the full tag string from a WDS sample."""
    if "json" in sample:
        data = sample["json"]
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))
        if isinstance(data, dict):
            # Build full prompt from structured tags
            parts = []
            if "tags" in data and isinstance(data["tags"], list):
                for tag_entry in data["tags"]:
                    if "tags" in tag_entry and isinstance(tag_entry["tags"], dict):
                        t = tag_entry["tags"]
                        for category in ["rating", "character", "general"]:
                            tag_list = t.get(category, [])
                            if isinstance(tag_list, list):
                                for item in tag_list:
                                    if isinstance(item, dict) and "name" in item:
                                        parts.append(str(item["name"]))
            else:
                # Fallback to old structure
                for key in ["rating", "character_tags", "general_tags"]:
                    tags = data.get(key, [])
                    if isinstance(tags, list):
                        parts.extend(str(t) for t in tags)
                    elif tags:
                        parts.append(str(tags))
            if parts:
                return " ".join(parts)[:512]
            return str(data)[:512]
        return str(data)[:512]

    if "txt" in sample:
        txt = sample["txt"]
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        return txt[:512]

    return ""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cache VAE latents grouped by aspect-ratio bucket")
    parser.add_argument("--input", type=str, required=True,
                        help="WebDataset URL/glob, e.g. '/data/shards/00{001..020}.tar'")
    parser.add_argument("--output_dir", type=str, default="./cached_latents",
                        help="Directory to write output .npz files")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Base image size for bucket calculation")
    parser.add_argument("--bucket_size", type=int, default=128,
                        help="Number of images per output file (matches training batch size)")
    parser.add_argument("--encode_batch_size", type=int, default=32,
                        help="VAE encoding batch size (VRAM dependent)")
    parser.add_argument("--vae_model", type=str, default="fal/FLUX.2-Tiny-AutoEncoder",
                        help="HuggingFace model ID for the Tiny VAE")
    parser.add_argument("--use_standard_vae", action="store_true",
                        help="Use standard FLUX.2 VAE instead of Tiny VAE")
    parser.add_argument("--stats_path", type=str, default="vae_stats.pt",
                        help="Path to vae_stats.pt for latent normalization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load VAE ────────────────────────────────────────────────────────────
    if args.use_standard_vae:
        from diffusers import AutoencoderKLFlux2
        print("Loading Standard FLUX.2 VAE...")
        vae = AutoencoderKLFlux2.from_pretrained(
            "black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch.bfloat16
        )
    else:
        from flux2_tiny_autoencoder import Flux2TinyAutoEncoder
        print(f"Loading Tiny VAE from {args.vae_model}...")
        vae = Flux2TinyAutoEncoder.from_pretrained(args.vae_model)
        
    vae = vae.to(device=device, dtype=torch.bfloat16).eval()

    # ── Load normalization stats ────────────────────────────────────────────
    if os.path.exists(args.stats_path):
        print(f"Loading VAE normalization stats from {args.stats_path}")
        stats = torch.load(args.stats_path, map_location="cpu")
        latents_mean = stats["mean"].to(device=device, dtype=torch.bfloat16)
        latents_std = stats["std"].to(device=device, dtype=torch.bfloat16)
    else:
        print(f"WARNING: {args.stats_path} not found — latents will NOT be normalized!")
        latents_mean = None
        latents_std = None

    # ── Prepare buckets ─────────────────────────────────────────────────────
    buckets = make_buckets(args.image_size)
    print(f"Buckets: {buckets}")

    # Accumulators: bucket_key -> {"tensors": [...], "prompts": [...]}
    bucket_data = {b: {"tensors": [], "prompts": []} for b in buckets}

    os.makedirs(args.output_dir, exist_ok=True)
    pattern = os.path.join(args.output_dir, "latents-%05d.tar")
    writer = wds.ShardWriter(pattern, maxsize=10**9) # 1GB shards

    # ── Encode and flush a full bucket ──────────────────────────────────────
    total_files_written = 0
    total_samples = 0

    def flush_bucket(bucket_key):
        nonlocal total_files_written, total_samples

        data = bucket_data[bucket_key]
        tensors = data["tensors"]
        prompts = data["prompts"]

        if not tensors:
            return

        bh, bw = bucket_key
        n = len(tensors)
        print(f"\n  Encoding bucket {bh}x{bw}: {n} images...")

        # Encode in sub-batches through the VAE
        all_latents = []
        for i in range(0, n, args.encode_batch_size):
            batch = torch.stack(tensors[i:i + args.encode_batch_size])
            batch = batch.to(device=device, dtype=torch.bfloat16)

            with torch.no_grad():
                if args.use_standard_vae:
                    import torch.nn.functional as F
                    latents = vae.encode(batch).latent_dist.mode()
                    # Reshape to 128ch for normalization since stats are in 2x2 patch format
                    latents = F.pixel_unshuffle(latents, 2)
                else:
                    out = vae.encode(batch, return_dict=False)
                    latents = out[0] if isinstance(out, tuple) else out

                if latents_mean is not None and latents_std is not None:
                    latents = (latents - latents_mean) / latents_std

                all_latents.append(latents.to(torch.float16).cpu())

        stacked = torch.cat(all_latents, dim=0).numpy()  # [N, 128, H', W']

        # Save to ShardWriter
        for idx in range(n):
            writer.write({
                "__key__": f"{total_samples + idx:09d}",
                "latent.npy": stacked[idx],
                "prompt.txt": prompts[idx]
            })

        total_samples += n
        total_files_written += 1 # We'll count bucket flushes just for logging

        # Reset accumulator
        data["tensors"].clear()
        data["prompts"].clear()

    # ── Open source dataset ─────────────────────────────────────────────────
    dataset = (
        wds.WebDataset(args.input, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
    )

    t0 = time.time()
    skipped = 0
    print("\nProcessing images...\n")

    for sample in tqdm(dataset, desc="Reading", unit="img"):
        # Find image
        image = None
        for ext in ["jpg", "png", "webp", "jpeg"]:
            if ext in sample:
                image = sample[ext]
                break

        if image is None or not isinstance(image, Image.Image):
            skipped += 1
            continue

        image = image.convert("RGB")
        w, h = image.size
        bucket_key = find_best_bucket(w, h, buckets)
        target_h, target_w = bucket_key

        try:
            tensor = preprocess_image(image, target_h, target_w)
        except Exception as e:
            print(f"  Skipping {sample.get('__key__', '?')}: {e}")
            skipped += 1
            continue

        prompt = extract_prompt(sample)
        bucket_data[bucket_key]["tensors"].append(tensor)
        bucket_data[bucket_key]["prompts"].append(prompt)

        # Flush when a bucket reaches bucket_size
        if len(bucket_data[bucket_key]["tensors"]) >= args.bucket_size:
            flush_bucket(bucket_key)

    # ── Flush remaining (partial buckets) ───────────────────────────────────
    print("\nFlushing remaining partial buckets...")
    for bk in buckets:
        if bucket_data[bk]["tensors"]:
            flush_bucket(bk)

    writer.close()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done!  {total_samples} latents written to tar shards.")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.1f}s  ({total_samples / max(elapsed, 1):.1f} img/s)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
