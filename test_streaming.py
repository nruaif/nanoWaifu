import argparse
import io
import tarfile
from pathlib import Path

import numpy as np


def new_stats():
    return {
        "n": 0,
        "sum": 0.0,
        "sum2": 0.0,
    }


def update_stats(x: np.ndarray, stats: dict):
    x = x.astype(np.float64, copy=False)
    stats["n"] += x.size
    stats["sum"] += x.sum()
    stats["sum2"] += np.square(x).sum()


def finalize_stats(stats: dict):
    if stats["n"] == 0:
        return None, None

    mean = stats["sum"] / stats["n"]
    var = (stats["sum2"] / stats["n"]) - mean**2
    std = np.sqrt(max(var, 0.0))

    return float(mean), float(std)


def iter_latents_from_shard(path: Path):
    with tarfile.open(path, "r:") as tar:
        found = False

        for member in tar:
            if not member.name.endswith("latent.npy"):
                continue

            found = True
            f = tar.extractfile(member)

            if f is None:
                continue

            latent = np.load(io.BytesIO(f.read()))

            if latent.ndim != 3:
                raise ValueError(
                    f"{path.name}:{member.name} expected shape [C,H,W], got {latent.shape}"
                )

            yield latent

        if not found:
            raise RuntimeError(f"{path.name}: no latent.npy entries found")


def check_shard(path: Path, mean_target: float, std_target: float, tol: float):
    stats = new_stats()
    samples = 0

    for latent in iter_latents_from_shard(path):
        update_stats(latent, stats)
        samples += 1

    mean, std = finalize_stats(stats)

    mean_ok = abs(mean - mean_target) <= tol
    std_ok = abs(std - std_target) <= tol
    normalized = mean_ok and std_ok

    return {
        "path": path,
        "samples": samples,
        "values": stats["n"],
        "mean": mean,
        "std": std,
        "mean_ok": mean_ok,
        "std_ok": std_ok,
        "normalized": normalized,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check whether latent shard tar files are normalized."
    )

    parser.add_argument("shards", nargs="+", help="Shard .tar files to check")
    parser.add_argument("--mean_target", type=float, default=0.0)
    parser.add_argument("--std_target", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=0.1)

    args = parser.parse_args()

    any_failed = False

    for shard in args.shards:
        path = Path(shard)

        try:
            result = check_shard(
                path=path,
                mean_target=args.mean_target,
                std_target=args.std_target,
                tol=args.tol,
            )

            print("=" * 80)
            print(f"Shard:      {result['path']}")
            print(f"Samples:    {result['samples']}")
            print(f"Values:     {result['values']}")
            print(f"Mean:       {result['mean']:.8f}")
            print(f"Std:        {result['std']:.8f}")
            print(f"Mean OK:    {result['mean_ok']}")
            print(f"Std OK:     {result['std_ok']}")
            print(f"Normalized: {result['normalized']}")

            if not result["normalized"]:
                any_failed = True

        except Exception as e:
            print("=" * 80)
            print(f"Shard:      {path}")
            print(f"ERROR:      {e}")
            any_failed = True

    raise SystemExit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
