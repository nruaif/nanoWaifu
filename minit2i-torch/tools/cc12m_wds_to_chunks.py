from __future__ import annotations

import argparse
from pathlib import Path

import torch
import webdataset as wds
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transform(image_size: int):
    # Matches the transform used everywhere else in the repo (see
    # mini_t2i/datasets/finetune.py): resize shorter side -> image_size with
    # BICUBIC, center crop to a square, keep uint8 pixels via PILToTensor.
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ]
    )


def to_sample(sample: dict, transform):
    image = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected decoded PIL image, got {type(image)!r}")
    caption = sample.get("txt", "")
    if isinstance(caption, bytes):
        caption = caption.decode("utf-8", errors="replace")
    return transform(image.convert("RGB")), str(caption)


def write_chunk(images, captions, out_dir: Path, chunk_idx: int, tokenizer, prompt_length: int) -> Path:
    tokens = tokenizer(
        captions,
        max_length=prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    payload = {
        "pixel_values": torch.stack(images, dim=0),
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "caption": captions,
    }
    path = out_dir / f"chunk_{chunk_idx:06d}.pt"
    # Write to a temp file then rename so an interrupted job never leaves a
    # half-written chunk that the loader would choke on.
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.rename(path)
    return path


def make_webdataset(shards: list[str]):
    try:
        return wds.WebDataset(
            shards,
            handler=wds.warn_and_continue,
            empty_check=False,
            shardshuffle=False,
        )
    except TypeError:
        return wds.WebDataset(shards, handler=wds.warn_and_continue, shardshuffle=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert img2dataset WebDataset shards into MiniT2I chunk_*.pt tensor chunks "
        "for the local_folder pretraining backend."
    )
    parser.add_argument("--wds", required=True, help="Directory of img2dataset .tar shards.")
    parser.add_argument("--out", required=True, help="Destination dir for chunk_*.pt (CC12M_TENSOR_ROOT).")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokenizer", default="google/flan-t5-large")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many samples (0 = all).")
    parser.add_argument("--start-chunk", type=int, default=0, help="First chunk index (for resuming).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(str(p) for p in Path(args.wds).glob("*.tar"))
    if not shards:
        raise SystemExit(f"no .tar shards found under {args.wds}")
    print(f"found {len(shards)} shard(s) under {args.wds}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=args.prompt_length)
    transform = build_transform(args.image_size)

    dataset = (
        make_webdataset(shards)
        .decode("pil", handler=wds.warn_and_continue)
        .map(lambda s: to_sample(s, transform), handler=wds.warn_and_continue)
    )

    images: list[torch.Tensor] = []
    captions: list[str] = []
    chunk_idx = args.start_chunk
    written = 0
    total = 0
    for image, caption in dataset:
        images.append(image)
        captions.append(caption)
        total += 1
        if len(images) == args.chunk_size:
            path = write_chunk(images, captions, out_dir, chunk_idx, tokenizer, args.prompt_length)
            print(f"wrote {path} ({total} samples)", flush=True)
            images, captions = [], []
            chunk_idx += 1
            written += 1
        if args.limit and total >= args.limit:
            break

    if images:
        path = write_chunk(images, captions, out_dir, chunk_idx, tokenizer, args.prompt_length)
        print(f"wrote {path} ({total} samples)", flush=True)
        written += 1

    print(f"done: {total} samples -> {written} chunks in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
