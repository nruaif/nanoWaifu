"""
Overfit test for the ConvNeXt FCDM backbone + EMF objective.

Trains a small (~2M parameter) FCDM with decoder attention to memorize a
single image (resized so its shortest side is 512, encoded to f/8 latents
with the FLUX.2 Tiny VAE) using the Euler Mean Flow loss (x1-prediction).
The model then reproduces the image in a single forward pass.

Every --log-every steps a 1-step EMF sample (left) and the VAE
reconstruction of the target latent (right) are saved to overfit_logs/.

Usage:
    python test_overfit.py --steps 5000
    python test_overfit.py --image path/to/img.jpg --size 512 --dim 32
"""
import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import trange

from emf import emf_loss, sample_emf_times
from flux2_tiny_autoencoder import Flux2TinyAutoEncoder, normalize_latent_f8
from model_dit import FCDM, TagProcessor

DEFAULT_IMAGE = (
    r"C:\Users\nRuaif\Downloads"
    r"\__hatsune_miku_artoria_pendragon_cirno_akemi_homura_kaname_madoka_and_97_more"
    r"_original_and_98_more_drawn_by_shinanashina__"
    r"cf7381d1616cbec451c495c37f0e7805(1).jpg"
)


def prepare_image(path, size):
    """Resize so the shortest side is `size`, then crop to a 16-multiple."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    w, h = img.size
    return img.crop((0, 0, (w // 16) * 16, (h // 16) * 16))


def encode_image(vae, img, device):
    """Encode a PIL image to a (normalized, if stats exist) f/8 latent."""
    x = TF.to_tensor(img).unsqueeze(0).to(device)
    x = x * 2.0 - 1.0  # [0,1] -> [-1,1]
    with torch.no_grad():
        out = vae.encode(x)
        z = out.latent if hasattr(out, "latent") else (out[0] if isinstance(out, tuple) else out)
        z = F.pixel_shuffle(z.float(), 2)  # [1, 32, H/8, W/8]
        if os.path.exists("vae_stats.pt"):
            stats = torch.load("vae_stats.pt", map_location="cpu")
            z = normalize_latent_f8(z, stats["mean"], stats["std"])
        else:
            print("Note: vae_stats.pt not found - training WITHOUT latent normalization.")
    return z[0]  # [32, H/8, W/8]


def decode_latent(vae, z):
    """Decode f/8 latents [N, 32, H, W] back to images in [0, 1]."""
    zp = F.pixel_unshuffle(z, 2)  # [N, 128, H/2, W/2] patchified for the Tiny VAE
    with torch.no_grad():
        out = vae.decode(zp)
        img = out.sample if hasattr(out, "sample") else (out[0] if isinstance(out, tuple) else out)
    return (img.clamp(-1, 1) + 1.0) / 2.0


def main():
    parser = argparse.ArgumentParser(description="FCDM + EMF single-image overfit test")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--size", type=int, default=512, help="target shortest side")
    parser.add_argument("--dim", type=int, default=32, help="FCDM base channels")
    parser.add_argument("--depth", type=int, default=2, help="FCDM depth (L)")
    parser.add_argument("--attn_every", type=int, default=3, help="decoder attention period (0=off)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4, help="copies of the image per step")
    parser.add_argument("--interval_ratio", type=float, default=0.5)
    parser.add_argument("--delta_t", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("overfit_logs", exist_ok=True)

    # -- Data: single image -> f/8 latent -------------------------------------
    img = prepare_image(args.image, args.size)
    print(f"Image: {img.size[0]}x{img.size[1]} (shortest side {args.size})")

    vae = Flux2TinyAutoEncoder.from_pretrained("fal/FLUX.2-Tiny-AutoEncoder").to(device).eval()
    z = encode_image(vae, img, device)
    print(f"Latent: {tuple(z.shape)} (f/8)")
    z_batch = z.unsqueeze(0).repeat(args.batch, 1, 1, 1)

    # -- Model -----------------------------------------------------------------
    model = FCDM(
        in_channels=z.shape[0],
        dim=args.dim,
        depth=args.depth,
        num_classes=1,
        attn_every=args.attn_every,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FCDM params: {n_params / 1e6:.2f}M (dim={args.dim}, depth={args.depth}, "
          f"attn_every={args.attn_every})")

    # -- Conditioning: a single tag, no CFG ------------------------------------
    with open("_overfit_tags.txt", "w", encoding="utf-8") as f:
        f.write("subject\n")
    tp = TagProcessor("_overfit_tags.txt")
    cond = tp.process_prompts(["subject"] * args.batch, device)
    cond_null = tp.process_prompts([""] * args.batch, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    fixed_noise = torch.randn(args.batch, *z.shape, device=device)

    print(f"Training EMF (x1-prediction) for {args.steps} steps...")
    for step in trange(args.steps, desc="Overfit", dynamic_ncols=True):
        t, r = sample_emf_times(args.batch, device, interval_ratio=args.interval_ratio)
        noise = torch.randn_like(z_batch)
        xt = (1 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * z_batch
        loss, _ = emf_loss(model, xt, z_batch, t, r, cond, cond_null,
                           delta_t=args.delta_t, cfg_scale=0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                ones = torch.ones(args.batch, device=device)
                endpoint = model(fixed_noise, torch.zeros(args.batch, device=device),
                                 cond[0], cond[1], r=ones)
                mse = F.mse_loss(endpoint, z_batch).item()
            sample = decode_latent(vae, endpoint[:1].float())
            target = decode_latent(vae, z_batch[:1].float())
            grid = torchvision.utils.make_grid(torch.cat([sample, target]), nrow=2)
            torchvision.utils.save_image(grid, f"overfit_logs/step_{step:05d}.png")
            print(f"  step {step:5d} | 1-step latent mse {mse:.4f}")

    os.remove("_overfit_tags.txt")
    print(f"\nDone. Check overfit_logs/ for sample (left) vs target (right) grids.")


if __name__ == "__main__":
    main()
