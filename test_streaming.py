import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
import math

# Import your model here (assuming you saved the previous code in model.py)
from model_dit import TokenformerDiT


# ==========================================
# 1. Image Processing & Patchification
# ==========================================
def load_and_preprocess_image(image_path, device):
    """Resizes shortest edge to 256, ensuring both edges are divisible by 32."""
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    # Scale shortest edge to 256
    scale = 256.0 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Ensure divisible by 32
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32

    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale to [-1, 1]
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"Loaded image. Original size: {w}x{h}. Resized to: {new_w}x{new_h}")
    return img_tensor


def patchify(x, p=16):
    """Converts image (B, C, H, W) to patches (B, C*p*p, H//p, W//p)"""
    B, C, H, W = x.shape
    Hp, Wp = H // p, W // p
    x = x.reshape(B, C, Hp, p, Wp, p)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * p * p, Hp, Wp)
    return x


def unpatchify(x, p=16):
    """Converts patches (B, C*p*p, H//p, W//p) back to image (B, C, H, W)"""
    B, C_pp, Hp, Wp = x.shape
    C = C_pp // (p * p)
    x = x.reshape(B, C, p, p, Hp, Wp)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C, Hp * p, Wp * p)
    return x


# ==========================================
# 2. Overfitting Script
# ==========================================
def train_overfit(image_path, iterations=200000, patch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    x1_img = load_and_preprocess_image(image_path, device)
    x1 = patchify(x1_img, p=patch_size)  # Shape: (1, 3*p*p, H/p, W/p)

    B, C, Hp, Wp = x1.shape

    # 2. Initialize Model
    model = TokenformerDiT(
        in_channels=3,
        base_channels=256,  # Scaled down for faster local testing
        num_blocks=4,
        heads=8,
        patch_size=patch_size,
        use_deco_decoder=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Create an output directory
    os.makedirs("overfit_outputs", exist_ok=True)
    save_image(x1_img * 0.5 + 0.5, "overfit_outputs/target_image.png")

    print(f"Starting overfit loop... Target patch grid: {Hp}x{Wp}")
    model.train()

    for step in range(iterations):
        optimizer.zero_grad()

        # --- Continuous Flow Matching Setup ---
        # 1. Sample continuous timestep t ~ U[0, 1]
        t = torch.rand((B,), device=device)
        t_condition = t * 1000  # Scale up to 0-1000 for the TimestepEmbedder

        # 2. Sample pure noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 3. Interpolate x_t
        t_expand = t.view(B, 1, 1, 1)
        x_t = t_expand * x1 + (1 - t_expand) * x0

        # 4. Target Velocity
        v_target = x1 - x0

        # --- Forward Pass ---
        # We pass unconditional label (-1 or None depending on your impl)
        v_pred, logvar_theta = model(x_t, t_condition, y=None)

        # --- Loss Calculation ---
        # 1. Standard Velocity MSE (reduced over channels, keeping spatial grid)
        mse_loss_raw = F.mse_loss(v_pred, v_target, reduction='none')
        mse_loss_spatial = mse_loss_raw.mean(dim=1)  # Shape: (B, Hp, Wp)

        # 2. Confidence / NLL Loss (Uncertainty learning)
        # Apply Stop-Gradient so variance head doesn't ruin velocity head gradients
        mse_loss_sg = mse_loss_spatial.detach()
        nll_loss = 0.5 * (mse_loss_sg * torch.exp(-logvar_theta) + logvar_theta)

        # 3. Total Combined Loss
        loss_v = mse_loss_spatial.mean()
        loss_nll = nll_loss.mean()
        loss = loss_v + loss_nll

        loss.backward()
        optimizer.step()

        # Logging & Sampling
        if step % 100 == 0:
            print(
                f"Step {step:04d} | Total Loss: {loss.item():.4f} | V-MSE: {loss_v.item():.4f} | NLL: {loss_nll.item():.4f}")

        if step > 0 and step % 500 == 0:
            model.eval()
            print(f"--> Generating sample at step {step}...")
            # Sample using the cascaded sampler logic
            sampled_patches = model.sample(
                B=1, H=Hp, W=Wp,
                device=device,
                maskgit_steps=16,  # Reduced for faster preview
                deco_steps=20,
                y=None
            )

            # Reconstruct image and save
            sampled_img = unpatchify(sampled_patches, p=patch_size)
            # Un-normalize from [-1, 1] to [0, 1] for saving
            sampled_img = torch.clamp(sampled_img * 0.5 + 0.5, 0, 1)
            save_image(sampled_img, f"overfit_outputs/sample_step_{step:04d}.png")
            model.train()

    print("Overfitting complete!")


if __name__ == "__main__":
    # Replace 'test_image.jpg' with a path to your local image
    image_file = "kurimi.jpg"

    if not os.path.exists(image_file):
        # Create a dummy image if one doesn't exist just to test the script pipeline
        print("Creating dummy test_image.jpg for testing...")
        dummy_img = torch.rand(3, 512, 768)
        save_image(dummy_img, image_file)

    train_overfit(image_file, iterations=3000)