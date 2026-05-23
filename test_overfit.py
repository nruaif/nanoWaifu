import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision
import os
from flux2_tiny_autoencoder import Flux2TinyAutoEncoder
from model_dit import TokenformerDiT, TagProcessor, sample_flow
from tqdm import tqdm


class DummyTagProcessor:
    """Minimal tag processor for overfitting tests with direct class index control."""
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def process_prompts(self, prompts, device):
        """Map prompts to class indices. Empty prompts map to the null class."""
        indices = []
        offsets = [0]
        for p in prompts:
            if p.strip():
                # Interpret prompt as an integer class index
                try:
                    indices.append(int(p))
                except ValueError:
                    indices.append(self.num_classes)  # null class
            else:
                indices.append(self.num_classes)  # null class for empty prompts
            offsets.append(len(indices))
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return indices, offsets

def prepare_image(image_path, size=256):
    img = Image.open(image_path).convert("RGB")
    # Simple center crop to maintain aspect ratio then resize
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = (w + min_side) // 2
    bottom = (h + min_side) // 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img

def main():
    torch._dynamo.config.disable = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    NUM_IMAGES = 8  # Number of images to load and overfit on

    # 1. Load images from D:\out2\goodness
    img_dir = r"D:\out2\goodness"
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} not found. Using local images if any.")
        # Fallback to current dir if D: is not available
        img_files = [f for f in os.listdir(".") if f.endswith((".jpg", ".png", ".webp"))][:NUM_IMAGES]
        if not img_files:
             print("No images found for overfitting.")
             return
    else:
        all_files = [f for f in os.listdir(img_dir) if f.endswith((".webp", ".jpg", ".png"))]
        img_files = [os.path.join(img_dir, f) for f in all_files[:NUM_IMAGES]]

    print(f"Loading {len(img_files)} images...")
    img_tensors = []
    for img_path in img_files:
        img = prepare_image(img_path)
        img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        img_tensor = (img_tensor - 0.5) * 2.0
        img_tensors.append(img_tensor)
    
    batch_imgs = torch.cat(img_tensors, dim=0)  # [N, 3, 256, 256]

    # 2. VAE and Latent Caching
    print("Loading Tiny VAE...")
    try:
        vae = Flux2TinyAutoEncoder.from_pretrained("fal/FLUX.2-Tiny-AutoEncoder").to(device).eval()
    except Exception as e:
        print(f"Could not load fal/FLUX.2-Tiny-AutoEncoder: {e}")
        vae = Flux2TinyAutoEncoder().to(device).eval()

    # Load normalization stats
    stats_path = "vae_stats.pt"
    if os.path.exists(stats_path):
        print(f"Loading VAE normalization stats from {stats_path}...")
        stats = torch.load(stats_path, map_location='cpu')
        latents_mean = stats["mean"].to(device, dtype=torch.float32)
        latents_std = stats["std"].to(device, dtype=torch.float32)
    else:
        print("Warning: vae_stats.pt not found. Running WITHOUT latent normalization.")
        latents_mean = None
        latents_std = None

    print("Encoding images to latents...")
    with torch.no_grad():
        out = vae.encode(batch_imgs.to(torch.float32))
        latents = out.latent # [8, 128, 16, 16]
        if latents_mean is not None and latents_std is not None:
            latents = (latents - latents_mean) / latents_std
            print("Normalized encoded latents using VAE running stats.")
        latents = F.pixel_shuffle(latents, 2)
    print(f"Latents shape: {latents.shape}")

    # 3. Model Setup
    # FCDM-L Model
    # in_channels=32 (f8 representation), dim=512 (FCDM-L specification)
    NUM_CLASSES = 8
    model_dit = TokenformerDiT(
        in_channels=32,
        dim=64,
        num_classes=NUM_CLASSES,
    ).to(device)

    print(f"DiT Params: {sum(p.numel() for p in model_dit.parameters() if p.requires_grad) / 1e6:.2f} M")

    # 4. Overfitting Loop
    opt_dit = torch.optim.AdamW(model_dit.parameters(), lr=5e-4)
    
    y_indices = torch.arange(NUM_IMAGES, device=device)
    y_offsets_dit = torch.arange(NUM_IMAGES + 1, device=device)

    os.makedirs("overfit_logs", exist_ok=True)

    print("Starting overfit training (1000 steps)...")
    for i in tqdm(range(100001)):
        # --- DiT Training ---
        model_dit.train()
        B, C, H, W = latents.shape
        t = torch.rand((B, ), device=device)
        t_reshaped = t.view(B, 1, 1, 1)
        noise = torch.randn_like(latents)
        
        # DiT uses (1-t)*x + t*noise
        xt_dit = (1 - t_reshaped) * latents + t_reshaped * noise
        v_target = noise - latents
        
        v_pred, match_loss = model_dit(xt_dit, t, y_indices, y_offsets_dit[:-1], return_layer_match=True)
        loss_dit = F.mse_loss(v_pred, v_target)
        
        total_loss = loss_dit + 0.2 * match_loss
        
        opt_dit.zero_grad()
        total_loss.backward()
        opt_dit.step()

        if i % 50 == 0:
            print(f"Step {i:4d} | DiT Loss: {loss_dit.item():.6f} | Match Loss: {match_loss.item():.6f}")
            
            # --- Sampling and Logging ---
            model_dit.eval()
            with torch.no_grad():
                # Sample DiT
                tp = DummyTagProcessor(NUM_CLASSES)
                # Use the actual training class labels as prompts
                sample_prompts = [str(c) for c in range(NUM_IMAGES)]
                samples_dit = sample_flow(
                    model_dit, tp, (H, W), NUM_IMAGES, sample_prompts, device, steps=50, sampler_type="euler"
                )
                
                samples_dit = F.pixel_unshuffle(samples_dit, 2)
                # De-normalize samples if stats are available
                if latents_mean is not None and latents_std is not None:
                    samples_dit = (samples_dit * latents_std) + latents_mean
                
                # Decode
                out_dit = vae.decode(samples_dit.to(torch.float32)).sample
                
                out_dit = (out_dit / 2.0 + 0.5).clamp(0, 1)
                
                # Create a grid: Top row DiT
                grid_img = torchvision.utils.make_grid(out_dit, nrow=2)
                torchvision.utils.save_image(grid_img, f"overfit_logs/step_{i:04d}.png")

    print("\nOverfitting complete. Check overfit_logs/ directory.")

if __name__ == "__main__":
    main()
