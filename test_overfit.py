import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import math
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torchvision.io as tv_io
from model_dit import MAE_TokenformerDiT


# ==========================================
# 4. Image Processing & Utilities
# ==========================================
def load_and_preprocess_image(image_path, device):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    scale = 256.0 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32

    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"Loaded image. Original size: {w}x{h}. Resized to: {new_w}x{new_h}")
    return img_tensor


def patchify(x, p=16):
    B, C, H, W = x.shape
    Hp, Wp = H // p, W // p
    x = x.reshape(B, C, Hp, p, Wp, p)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * p * p, Hp, Wp)
    return x


def unpatchify(x, p=16):
    B, C_pp, Hp, Wp = x.shape
    C = C_pp // (p * p)
    x = x.reshape(B, C, p, p, Hp, Wp)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C, Hp * p, Wp * p)
    return x


# ==========================================
# 5. Overfitting Script
# ==========================================
def train_overfit(image_path, iterations=200000, patch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x1_img = load_and_preprocess_image(image_path, device)
    x1 = patchify(x1_img, p=patch_size)

    B, C, Hp, Wp = x1.shape
    H_img, W_img = x1_img.shape[-2], x1_img.shape[-1]  # For upsampling the visualization

    model = MAE_TokenformerDiT(
        in_channels=3,
        base_channels=256,
        num_blocks=4,
        heads=8,
        patch_size=patch_size,
        use_deco_decoder=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    os.makedirs("overfit_outputs", exist_ok=True)
    save_image(x1_img * 0.5 + 0.5, "overfit_outputs/target_image.png")

    print(f"Starting MAE overfit loop... Target patch grid: {Hp}x{Wp}")
    model.train()

    # --- NEW: List to store our frames for the video ---
    logvar_frames = []

    for step in range(iterations):
        optimizer.zero_grad()

        t = torch.empty((B,), device=device).uniform_(0, 1.0)
        t_condition = t * 1000

        x0 = torch.randn_like(x1)

        t_expand = t.view(B, 1, 1, 1)
        x_t = t_expand * x1 + (1 - t_expand) * x0
        seq_len = Hp * Wp
        mask_ratio = torch.normal(mean=0.75, std=0.25, size=(B,), device=device)
        mask_ratio = torch.clamp(mask_ratio, min=0.0, max=1.0)
        n_masked = (mask_ratio * seq_len).long()
        mask = torch.zeros((B, seq_len), dtype=torch.bool, device=device)
        for b in range(B):
            if n_masked[b] > 0:
                perm = torch.randperm(seq_len, device=device)
                mask[b, perm[:n_masked[b]]] = True

        # Forward DiT on the CLEAN image (x1) to extract global context
        t_dit = torch.zeros((B,), device=device)
        c_low_freq, conf, c_dec_embed = model.forward_dit(x1, t_dit, y=None, mask=mask)

        # Forward Pixel Decoder on the NOISY image (x_t)
        t_emb = model.t_embedder(t_condition)
        t_emb_dec = model.c_enc_to_dec(t_emb)  # Project to MAE decoder dimension
        x_raw = x_t.flatten(2).transpose(1, 2)
        x1_pred_flat = model.pixel_decoder(x_raw, c_low_freq, Hp, Wp, t_emb=t_emb_dec)
        x1_pred = x1_pred_flat.transpose(1, 2).reshape(B, -1, Hp, Wp)

        # Extract confidence scores
        logvar_theta = conf.reshape(B, Hp, Wp)

        # Loss Calculation (in v-space)
        mse_loss_raw = F.mse_loss(x1_pred, x1, reduction='none')
        mse_loss_spatial = mse_loss_raw.mean(dim=1)

        mse_loss_sg = mse_loss_spatial.detach()
        nll_loss = 0.5 * (mse_loss_sg * torch.exp(-logvar_theta) + logvar_theta)

        loss_v = mse_loss_spatial.mean()
        loss_nll = nll_loss.mean()
        loss = loss_v + loss_nll * 0.01

        loss.backward()
        optimizer.step()

        # --- NEW: Capture logvar_theta for visualization every 20 steps ---
        if step % 20 == 0:
            with torch.no_grad():
                # Extract the 2D map for the first item in batch
                var_map = logvar_theta[0].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, Hp, Wp)

                # Upsample to match original image dimensions for a better-looking video
                var_map_up = F.interpolate(var_map, size=(H_img, W_img), mode='bilinear', align_corners=False).squeeze()

                # Normalize to [0, 1] range based on current min/max
                var_min, var_max = var_map_up.min(), var_map_up.max()
                if var_max > var_min:
                    var_norm = (var_map_up - var_min) / (var_max - var_min)
                else:
                    var_norm = var_map_up - var_min

                # Apply a colormap (viridis: purple=low variance/high confidence, yellow=high variance/low conf)
                cmap = plt.get_cmap('viridis')
                var_colored = cmap(var_norm.cpu().numpy())[..., :3]  # Drop Alpha channel

                # Convert to uint8 RGB tensor [H, W, 3] and append to frames
                var_colored_uint8 = (var_colored * 255).astype(np.uint8)
                logvar_frames.append(torch.from_numpy(var_colored_uint8))

        if step % 100 == 0:
            print(
                f"Step {step:04d} | Total Loss: {loss.item():.4f} | V-MSE: {loss_v.item():.4f} | NLL: {loss_nll.item():.4f}")

        if step > 0 and step % 500 == 0:
            model.eval()
            print(f"--> Generating MAE sample at step {step}...")
            sampled_patches = model.sample(
                B=1, H=Hp, W=Wp,
                device=device,
                maskgit_steps=25,
                deco_steps=50,
                y=None
            )

            sampled_img = unpatchify(sampled_patches, p=patch_size)
            sampled_img = torch.clamp(sampled_img * 0.5 + 0.5, 0, 1)
            save_image(sampled_img, f"overfit_outputs/sample_step_{step:04d}.png")
            model.train()

    print("Overfitting complete!")

    # --- NEW: Save the collected frames as a video ---
    if len(logvar_frames) > 0:
        print("Compiling logvar_theta evolution video...")
        # Stack frames into [T, H, W, C] format required by torchvision
        video_tensor = torch.stack(logvar_frames)
        video_path = "overfit_outputs/logvar_evolution.mp4"
        # Write out at 30 fps
        tv_io.write_file(video_path, video_tensor)
        print(f"Saved visualization video to: {video_path}")


if __name__ == "__main__":
    image_file = r"C:\Users\nRuaif\Downloads\__hatsune_miku_artoria_pendragon_cirno_akemi_homura_kaname_madoka_and_97_more_original_and_98_more_drawn_by_shinanashina__cf7381d1616cbec451c495c37f0e7805.jpg"

    if not os.path.exists(image_file):
        print("Creating dummy kurimi.jpg for testing...")
        dummy_img = torch.rand(3, 512, 768)
        save_image(dummy_img, image_file)

    train_overfit(image_file, iterations=300000)