"""
Normalize cached latent tars using VAE stats (vae_stats.pt) or runtime statistics.

If your cached latents were created without --stats_path (or with --stats_path missing),
they are **unnormalized** raw VAE outputs (128ch, after pixel_unshuffle for standard VAE).
This script converts them to normalized form:  (latent - mean) / std

Latents may not match vae_stats.pt (e.g., different VAE, different dataset bucket stats,
or latents already divided by 1.7 as in patched train.py). Use --runtime-stats to
compute per-channel mean/std directly from the input tars at runtime instead of
relying on vae_stats.pt.

Usage:
    # Using vae_stats.pt (original)
    python normalize_tar_latent.py --input "cached_latents/latents-*.tar" \
                                   --output_dir cached_latents_norm \
                                   --stats_path vae_stats.pt

    # Compute mean/std from data itself (recommended if vae_stats mismatch)
    python normalize_tar_latent.py --input "cached_latents/latents-*.tar" \
                                   --output_dir cached_latents_norm \
                                   --runtime-stats --save-stats runtime_stats.pt

    # Single file with auto-fallback: if vae_stats.pt missing -> compute runtime
    python normalize_tar_latent.py --input "C:/Downloads/00001.tar" --output_dir ./fixed --runtime-stats

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


class SimpleTarShardWriter:
    """
    Windows-safe fallback for webdataset.ShardWriter (which uses gopen and fails on 'C:\\' paths).
    Writes shards as plain tar files using tarfile module, sharding on maxcount/maxsize.
    """
    def __init__(self, pattern, maxcount=100000, maxsize=10**9, start_shard=0):
        self.pattern = pattern
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.shard_idx = start_shard
        self.count = 0
        self.size = 0
        self.tar = None
        self._open_next()

    def _open_next(self):
        if self.tar is not None:
            self.tar.close()
        fname = self.pattern % self.shard_idx
        os.makedirs(os.path.dirname(os.path.abspath(fname)) if os.path.dirname(fname) else ".", exist_ok=True)
        self.tar = tarfile.open(fname, "w")
        # print(f"Opened shard {fname}")
        self.shard_idx += 1
        self.count = 0
        self.size = 0

    def write(self, sample):
        key = sample.get("__key__", uuid.uuid4().hex)
        # Determine entries
        entries = []
        if "latent.npy" in sample:
            arr = sample["latent.npy"]
            # Ensure numpy array
            if isinstance(arr, torch.Tensor):
                arr = arr.numpy()
            buf = io.BytesIO()
            np.save(buf, arr)
            data = buf.getvalue()
            entries.append((f"{key}.latent.npy", data))
        if "prompt.txt" in sample:
            txt = sample["prompt.txt"]
            if isinstance(txt, str):
                txt = txt.encode("utf-8")
            entries.append((f"{key}.prompt.txt", txt))
        # Check if need to roll to next shard
        est_size = sum(len(d) for _, d in entries) + 1024  # overhead
        if self.count >= self.maxcount or (self.size + est_size) > self.maxsize:
            self._open_next()
        for name, data in entries:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            self.tar.addfile(ti, io.BytesIO(data))
        self.count += 1
        self.size += est_size

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def _get_writer(pattern, maxcount, maxsize, start_shard):
    """
    Try wds.ShardWriter, fallback to SimpleTarShardWriter on Windows/path issues.
    """
    try:
        # wds.ShardWriter will fail on Windows drive-letter paths due to gopen urlparse
        # Try to detect and force fallback on Windows with ':' in pattern
        if os.name == "nt" and ":" in pattern:
            raise ValueError("Windows drive letter path, use SimpleTarShardWriter")
        return wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize, start_shard=start_shard)
    except Exception as e:
        print(f"  Using SimpleTarShardWriter fallback due to: {e}")
        return SimpleTarShardWriter(pattern, maxcount=maxcount, maxsize=maxsize, start_shard=start_shard)


def load_stats(stats_path, device="cpu"):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"vae_stats.pt not found at {stats_path}. "
                                f"Generate it via cache_latents.py or train.py first run. "
                                f"Or use --runtime-stats to compute from data.")
    stats = torch.load(stats_path, map_location=device)
    # Support both {"mean":..., "std":...} and plain tensor cases
    if isinstance(stats, dict):
        mean = stats["mean"] if "mean" in stats else stats.get("running_mean")
        std = stats["std"] if "std" in stats else stats.get("running_var", None)
        if std is not None and "running_var" in stats:
            # running_var -> std
            import math
            eps = stats.get("eps", 1e-5) if isinstance(stats, dict) else 1e-5
            std = torch.sqrt(std + eps)
    else:
        raise ValueError(f"Unknown stats format in {stats_path}: {type(stats)}")
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


def _iter_tar_samples_tarfile(tar_path):
    """
    Fallback iterator using tarfile directly (Windows-safe, no wds gopen).
    Groups entries by prefix before first '.' as WebDataset does.
    Yields dict with '__key__', 'latent.npy' (bytes), 'prompt.txt' (bytes/str).
    """
    try:
        with tarfile.open(tar_path, "r") as tf:
            # Group by key
            grouped = {}
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name
                # WebDataset key is prefix before first '.'
                # e.g., "abc123.latent.npy" -> key "abc123", ext "latent.npy"
                if "." not in name:
                    continue
                key, ext = name.split(".", 1)
                # ext is like "latent.npy" or "prompt.txt"
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                if key not in grouped:
                    grouped[key] = {"__key__": key}
                grouped[key][ext] = data
            for sample in grouped.values():
                yield sample
    except Exception as e:
        print(f"  tarfile fallback failed for {tar_path}: {e}")
        return


def _iter_tar_samples(input_path):
    """
    Try wds first (Linux), fallback to tarfile on failure (Windows path with drive letter).
    """
    # Try wds if path looks like URL without drive colon issue
    # On Windows, wds.WebDataset fails for "C:\\..." due to urlparse; fallback to tarfile
    if os.name == "nt" and ":" in input_path:
        yield from _iter_tar_samples_tarfile(input_path)
        return
    try:
        dataset = wds.WebDataset(input_path, handler=wds.warn_and_continue)
        for sample in dataset:
            yield sample
        return
    except Exception as e:
        # Fallback
        yield from _iter_tar_samples_tarfile(input_path)


def compute_runtime_stats(input_files, max_samples=None, device="cpu"):
    """
    Compute per-channel mean/std directly from latent tars at runtime.
    Streams over all tars, accumulates per-channel sum/sumsq over all spatial positions.

    Returns mean, std as [1, C, 1, 1] float tensors.
    """
    print(f"Computing runtime per-channel stats from {len(input_files)} tar(s)...")
    # First pass: need to know C. Peek first sample.
    C = None
    sum_c = None
    sumsq_c = None
    total_pixels_per_channel = 0  # sum(H*W) over samples, per channel same count
    n_samples = 0

    for input_path in tqdm(input_files, desc="Scanning tars for stats"):
        for sample in _iter_tar_samples(input_path):
            if "latent.npy" not in sample:
                continue
            latent_data = sample["latent.npy"]
            if isinstance(latent_data, bytes):
                latent_np = np.load(io.BytesIO(latent_data))
            elif isinstance(latent_data, np.ndarray):
                latent_np = latent_data
            else:
                latent_np = np.array(latent_data)

            # latent_np: [C, H, W]
            if latent_np.ndim != 3:
                print(f"  Warning: unexpected latent shape {latent_np.shape}, skipping")
                continue
            c, h, w = latent_np.shape
            if C is None:
                C = c
                sum_c = np.zeros(C, dtype=np.float64)
                sumsq_c = np.zeros(C, dtype=np.float64)
                print(f"  Detected C={C}, example H={h} W={w}")
            elif c != C:
                print(f"  Warning: channel mismatch {c} vs {C}, skipping sample")
                continue

            # Accumulate per-channel sum over H*W
            # Use float64 for numerical stability
            sum_c += latent_np.astype(np.float64).sum(axis=(1, 2))
            sumsq_c += (latent_np.astype(np.float64) ** 2).sum(axis=(1, 2))
            total_pixels_per_channel += h * w
            n_samples += 1

            if max_samples is not None and n_samples >= max_samples:
                break
        if max_samples is not None and n_samples >= max_samples:
            break

    if n_samples == 0:
        raise RuntimeError("No latent.npy samples found for runtime stats computation.")

    # Compute mean/std
    mean_c = sum_c / total_pixels_per_channel
    var_c = (sumsq_c / total_pixels_per_channel) - (mean_c ** 2)
    var_c = np.maximum(var_c, 1e-12)  # clamp
    std_c = np.sqrt(var_c)

    print(f"Runtime stats computed over {n_samples} samples, {total_pixels_per_channel} pixels per channel:")
    print(f"  mean per-channel sample {mean_c[:5].tolist()} ... mean overall {float(mean_c.mean()):.4f}")
    print(f"  std per-channel sample  {std_c[:5].tolist()} ... mean overall {float(std_c.mean()):.4f}")
    print(f"  Checking if data looks already normalized: overall mean {float(mean_c.mean()):.4f} std {float(std_c.mean()):.4f}")
    if abs(float(mean_c.mean())) < 0.2 and abs(float(std_c.mean()) - 1.0) < 0.2:
        print("  -> Data appears ALREADY normalized (mean~0, std~1). Re-normalizing would distort. Consider --skip-if-normalized or not running this script.")

    mean = torch.from_numpy(mean_c).view(1, -1, 1, 1).float()
    std = torch.from_numpy(std_c).view(1, -1, 1, 1).float()
    return mean, std


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
    count = 0
    pbar = tqdm(desc=f"Normalizing {os.path.basename(input_path)}", unit="sample")
    for sample in _iter_tar_samples(input_path):
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
    parser = argparse.ArgumentParser(description="Normalize unnormalized cached latent tars via vae_stats.pt or runtime stats")
    parser.add_argument("--input", type=str, required=True,
                        help="Input tar glob, e.g. 'cached_latents/latents-*.tar' or single file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for normalized tars")
    parser.add_argument("--stats_path", type=str, default="vae_stats.pt",
                        help="Path to vae_stats.pt (from train.py or cache_latents.py). Ignored if --runtime-stats is set.")
    parser.add_argument("--runtime-stats", action="store_true",
                        help="Compute per-channel mean/std directly from input tars at runtime, instead of using vae_stats.pt. "
                             "Use this when vae_stats.pt does not match your data (e.g., different VAE, patched /1.7 scaling).")
    parser.add_argument("--runtime-max-samples", type=int, default=None,
                        help="Max samples to use for runtime stats estimation (default: all). For quick estimate on large datasets.")
    parser.add_argument("--save-stats", type=str, default=None,
                        help="If set, save computed/runtime stats to this path (e.g., runtime_vae_stats.pt) for reuse in train sampling.")
    parser.add_argument("--compare-stats", action="store_true",
                        help="If both vae_stats.pt and --runtime-stats are available, compare them and warn on mismatch.")
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

    # Determine stats source
    mean, std = None, None
    runtime_mean, runtime_std = None, None

    if args.runtime_stats:
        runtime_mean, runtime_std = compute_runtime_stats(input_files, max_samples=args.runtime_max_samples)
        mean, std = runtime_mean, runtime_std
        if args.save_stats:
            torch.save({"mean": mean, "std": std}, args.save_stats)
            print(f"Saved runtime stats to {args.save_stats} (use as --stats_path in train.py sampling)")

        if args.compare_stats and os.path.exists(args.stats_path):
            try:
                vae_mean, vae_std = load_stats(args.stats_path)
                # Compare
                mean_diff = (runtime_mean - vae_mean).abs().mean().item()
                std_diff = (runtime_std - vae_std).abs().mean().item()
                print(f"Comparison vae_stats vs runtime: mean L1 diff {mean_diff:.4f}, std L1 diff {std_diff:.4f}")
                if mean_diff > 0.2 or std_diff > 0.2:
                    print(f"  WARNING: vae_stats.pt does NOT match runtime data stats! Use --runtime-stats output for correct normalization.")
                    print(f"  vae_stats mean {vae_mean.flatten()[:3].tolist()} vs runtime {runtime_mean.flatten()[:3].tolist()}")
            except Exception as e:
                print(f"  Could not compare stats: {e}")
    else:
        # Try vae_stats.pt, fallback to runtime if missing
        if os.path.exists(args.stats_path):
            mean, std = load_stats(args.stats_path)
            # Optionally compute runtime for comparison if requested
            if args.compare_stats:
                runtime_mean, runtime_std = compute_runtime_stats(input_files, max_samples=args.runtime_max_samples or 5000)
                mean_diff = (runtime_mean - mean).abs().mean().item()
                std_diff = (runtime_std - std).abs().mean().item()
                print(f"Comparison vae_stats vs runtime (sample {args.runtime_max_samples or 5000}): mean diff {mean_diff:.4f}, std diff {std_diff:.4f}")
                if mean_diff > 0.2 or std_diff > 0.2:
                    print(f"  WARNING: vae_stats mismatch! Consider re-running with --runtime-stats for correct normalization.")
        else:
            print(f"vae_stats.pt not found at {args.stats_path}, falling back to --runtime-stats computation...")
            runtime_mean, runtime_std = compute_runtime_stats(input_files, max_samples=args.runtime_max_samples)
            mean, std = runtime_mean, runtime_std
            if args.save_stats:
                torch.save({"mean": mean, "std": std}, args.save_stats)
                print(f"Saved runtime stats to {args.save_stats}")

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

    writer = _get_writer(pattern, args.maxcount, args.maxsize, start_shard)

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
