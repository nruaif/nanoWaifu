#!/usr/bin/env python3
"""
compute_runtime_stats.py

Streaming calculation of per-channel mean and standard deviation from WebDataset tar shards.
Saves output as a PyTorch checkpoint dictionary:
    {"mean": torch.Tensor [C], "std": torch.Tensor [C]}
compatible with `train.py`'s `runtime_stats_online.pt`.

Features:
- Pure streaming: extracts numpy latents directly from tar bytes in memory (no disk extraction).
- Supports single tar, glob patterns (e.g., 'data/*.tar'), and brace expansions (e.g., 'latents-00{00..184}.tar').
- Numerically stable accumulation in float64.
- Optional multi-worker parallelism across tar shards.
- Detailed statistics printout & normalization check.

Usage:
    python compute_runtime_stats.py --input "Good_latents_normal_flux2/latents-*.tar"
    python compute_runtime_stats.py --input "/workspace/data/latents-00{00..184}.tar" --output runtime_stats_online.pt
    python compute_runtime_stats.py --input "data.tar" --num_workers 4 --max_samples 10000
"""

import argparse
import glob
import io
import os
import re
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm


def expand_brace_pattern(pattern: str) -> List[str]:
    """
    Expands bash-style brace patterns like 'latents-00{00..184}.tar' or 'data/{a,b,c}.tar'.
    Falls back to braceexpand library if installed, or regex generator.
    """
    try:
        import braceexpand
        return list(braceexpand.braceexpand(pattern))
    except ImportError:
        pass

    # Simple regex fallback for numeric ranges: {00..66} or {0..184}
    match = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if match:
        start_str, end_str = match.group(1), match.group(2)
        start, end = int(start_str), int(end_str)
        width = len(start_str) if start_str.startswith("0") else 0
        expanded = []
        for i in range(start, end + 1):
            num_str = f"{i:0{width}d}" if width > 0 else str(i)
            expanded.append(pattern[:match.start()] + num_str + pattern[match.end():])
        return expanded

    # Comma separated lists: {a,b,c}
    match_comma = re.search(r"\{([^}]+)\}", pattern)
    if match_comma:
        items = match_comma.group(1).split(",")
        expanded = []
        for it in items:
            expanded.append(pattern[:match_comma.start()] + it.strip() + pattern[match_comma.end():])
        return expanded

    return [pattern]


def resolve_input_paths(input_spec: str) -> List[str]:
    """
    Resolves input specification to a sorted list of existing .tar files.
    Supports comma-separated paths, brace expansion, globbing, and directories.
    """
    raw_patterns = [p.strip() for p in input_spec.split(",") if p.strip()]
    candidate_patterns = []
    for p in raw_patterns:
        candidate_patterns.extend(expand_brace_pattern(p))

    matched_files = set()
    for p in candidate_patterns:
        if os.path.isdir(p):
            for f in Path(p).glob("*.tar"):
                matched_files.add(str(f))
        elif os.path.isfile(p):
            matched_files.add(str(Path(p)))
        else:
            globbed = glob.glob(p)
            for f in globbed:
                if os.path.isfile(f):
                    matched_files.add(str(Path(f)))

    files = sorted(list(matched_files))
    return files


def process_single_tar(tar_path: str, max_samples_per_tar: Optional[int] = None) -> Dict:
    """
    Streams a single tar archive and accumulates sum and sum-of-squares per channel.
    Returns partial sums and sample counts.
    """
    sum_c: Optional[np.ndarray] = None
    sumsq_c: Optional[np.ndarray] = None
    total_pixels: int = 0
    n_samples: int = 0
    num_channels: Optional[int] = None
    min_val: float = float("inf")
    max_val: float = float("-inf")

    try:
        with tarfile.open(tar_path, "r:*") as tf:
            for member in tf:
                if not member.isfile():
                    continue

                name = member.name
                # Check for latent files (.latent.npy, latent.npy, or .npy)
                if not (name.endswith(".latent.npy") or name.endswith(".npy") or "latent" in name.lower()):
                    continue

                extracted = tf.extractfile(member)
                if extracted is None:
                    continue

                data_bytes = extracted.read()
                try:
                    arr = np.load(io.BytesIO(data_bytes))
                except Exception:
                    continue

                # Ensure shape is [C, H, W]
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.ndim == 2:
                    arr = arr[np.newaxis, ...]

                if arr.ndim != 3:
                    continue

                c, h, w = arr.shape
                if num_channels is None:
                    num_channels = c
                    sum_c = np.zeros(c, dtype=np.float64)
                    sumsq_c = np.zeros(c, dtype=np.float64)
                elif c != num_channels:
                    # Ignore samples with mismatched channel count
                    continue

                arr_f64 = arr.astype(np.float64)
                sum_c += arr_f64.sum(axis=(1, 2))
                sumsq_c += (arr_f64 ** 2).sum(axis=(1, 2))
                total_pixels += h * w
                n_samples += 1

                s_min = float(arr.min())
                s_max = float(arr.max())
                if s_min < min_val:
                    min_val = s_min
                if s_max > max_val:
                    max_val = s_max

                if max_samples_per_tar is not None and n_samples >= max_samples_per_tar:
                    break

    except Exception as e:
        print(f"[Warning] Failed reading {tar_path}: {e}")

    return {
        "num_channels": num_channels,
        "n_samples": n_samples,
        "total_pixels": total_pixels,
        "sum_c": sum_c,
        "sumsq_c": sumsq_c,
        "min_val": min_val if n_samples > 0 else 0.0,
        "max_val": max_val if n_samples > 0 else 0.0,
    }


def compute_runtime_stats(
    input_spec: str,
    output_path: str = "runtime_stats_online.pt",
    max_samples: Optional[int] = None,
    num_workers: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-channel mean and std from webdataset tar files.
    Saves to output_path and returns (mean, std) tensors.
    """
    files = resolve_input_paths(input_spec)
    if not files:
        raise FileNotFoundError(f"No tar files found matching specification: {input_spec}")

    print(f"\n=======================================================")
    print(f" Computing Online Runtime Latent Statistics")
    print(f"=======================================================")
    print(f"Found {len(files)} tar file(s). First few:")
    for f in files[:3]:
        print(f"  - {f}")
    if len(files) > 3:
        print(f"  ... and {len(files) - 3} more")
    print(f"Output path: {output_path}")
    print(f"Num workers: {num_workers}")
    print(f"Max samples limit: {max_samples if max_samples else 'None (all)'}\n")

    global_channels = None
    global_sum_c = None
    global_sumsq_c = None
    global_total_pixels = 0
    global_samples = 0
    overall_min = float("inf")
    overall_max = float("-inf")

    # Sequential processing
    if num_workers <= 1 or len(files) == 1:
        pbar = tqdm(files, desc="Scanning tar shards", unit="tar")
        for tar_path in pbar:
            rem_samples = (max_samples - global_samples) if max_samples is not None else None
            res = process_single_tar(tar_path, max_samples_per_tar=rem_samples)
            if res["n_samples"] == 0:
                continue

            c = res["num_channels"]
            if global_channels is None:
                global_channels = c
                global_sum_c = np.zeros(c, dtype=np.float64)
                global_sumsq_c = np.zeros(c, dtype=np.float64)
            elif c != global_channels:
                continue

            global_sum_c += res["sum_c"]
            global_sumsq_c += res["sumsq_c"]
            global_total_pixels += res["total_pixels"]
            global_samples += res["n_samples"]
            overall_min = min(overall_min, res["min_val"])
            overall_max = max(overall_max, res["max_val"])

            pbar.set_postfix({"samples": global_samples, "pixels_M": f"{global_total_pixels / 1e6:.2f}"})

            if max_samples is not None and global_samples >= max_samples:
                break
    else:
        # Parallel worker processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_tar = {executor.submit(process_single_tar, tar_p): tar_p for tar_p in files}
            pbar = tqdm(as_completed(future_to_tar), total=len(files), desc="Processing shards", unit="tar")
            for future in pbar:
                res = future.result()
                if res["n_samples"] == 0:
                    continue

                c = res["num_channels"]
                if global_channels is None:
                    global_channels = c
                    global_sum_c = np.zeros(c, dtype=np.float64)
                    global_sumsq_c = np.zeros(c, dtype=np.float64)
                elif c != global_channels:
                    continue

                global_sum_c += res["sum_c"]
                global_sumsq_c += res["sumsq_c"]
                global_total_pixels += res["total_pixels"]
                global_samples += res["n_samples"]
                overall_min = min(overall_min, res["min_val"])
                overall_max = max(overall_max, res["max_val"])

                pbar.set_postfix({"samples": global_samples, "pixels_M": f"{global_total_pixels / 1e6:.2f}"})

                if max_samples is not None and global_samples >= max_samples:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    if global_samples == 0 or global_total_pixels == 0:
        raise RuntimeError(f"No valid latent numpy samples found in the provided tar files.")

    # Calculate per-channel mean and std
    mean_np = global_sum_c / global_total_pixels
    var_np = (global_sumsq_c / global_total_pixels) - (mean_np ** 2)
    var_np = np.maximum(var_np, 1e-12)
    std_np = np.sqrt(var_np)

    mean_tensor = torch.from_numpy(mean_np.astype(np.float32))
    std_tensor = torch.from_numpy(std_np.astype(np.float32))

    # Save dictionary
    out_dict = {
        "mean": mean_tensor,
        "std": std_tensor,
        "num_channels": global_channels,
        "num_samples": global_samples,
        "total_pixels_per_channel": global_total_pixels,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, out_file)

    # Print summary diagnostics
    print(f"\n=======================================================")
    print(f" Statistics Summary ({global_samples} samples, {global_channels} channels)")
    print(f"=======================================================")
    print(f"Total pixels per channel: {global_total_pixels:,}")
    print(f"Data value dynamic range: [{overall_min:.4f}, {overall_max:.4f}]")
    print(f"Per-channel Mean: min={mean_np.min():.5f}, max={mean_np.max():.5f}, avg={mean_np.mean():.5f}")
    print(f"Per-channel Std:  min={std_np.min():.5f}, max={std_np.max():.5f}, avg={std_np.mean():.5f}")
    print(f"First 5 channel means: {[round(float(x), 4) for x in mean_np[:5]]}")
    print(f"First 5 channel stds:  {[round(float(x), 4) for x in std_np[:5]]}")

    # Heuristic check
    if abs(float(mean_np.mean())) < 0.1 and abs(float(std_np.mean()) - 1.0) < 0.15:
        print("\n[Notice] Latents already appear normalized (mean ~ 0, std ~ 1).")
    else:
        print(f"\n[Info] Latents require normalization: data_mean ~ {mean_np.mean():.4f}, data_std ~ {std_np.mean():.4f}.")

    print(f"\nSuccessfully saved runtime stats to -> {out_file.resolve()}")
    return mean_tensor, std_tensor


def main():
    parser = argparse.ArgumentParser(description="Calculate runtime_stats_online.pt from WebDataset tar shards.")
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input path specification: single tar file, glob ('data/*.tar'), brace pattern ('latents-00{00..184}.tar'), or directory."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="runtime_stats_online.pt",
        help="Path to save output PyTorch stats dictionary (default: 'runtime_stats_online.pt')."
    )
    parser.add_argument(
        "--max_samples", "-n", type=int, default=None,
        help="Maximum number of latent samples to accumulate across all shards."
    )
    parser.add_argument(
        "--num_workers", "-w", type=int, default=1,
        help="Number of parallel worker processes for scanning multiple tar files (default: 1)."
    )

    args = parser.parse_args()
    compute_runtime_stats(
        input_spec=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
