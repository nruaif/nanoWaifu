import torch
import torch.nn.functional as F
import yaml
import os
import glob
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from train import TagProcessor
from model_dit import PixNerDiT
import numpy as np
import matplotlib.pyplot as plt

def get_latest_checkpoint(ckpt_dir):
    checkpoints = glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pth"))
    if not checkpoints:
        return None
    # Sort by step number
    return sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

def main():
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('activations', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    tag_processor = TagProcessor("tags.txt")
    num_classes = tag_processor.num_classes
    pad_idx = tag_processor.pad_idx
    txt_max_length = tag_processor.max_tags
    
    print("Initializing model...")
    model = PixNerDiT(
        in_channels=3,
        hidden_size=1024,
        num_groups=16,
        patch_size=16,
        vocab_size=num_classes,
        txt_max_length=txt_max_length,
    ).to(device)
    
    # Load checkpoint
    ckpt_dir = config.get('resume_from', 'outputs_dit/')
    ckpt_path = get_latest_checkpoint(ckpt_dir)
    
    if ckpt_path:
        print(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
    else:
        print("No checkpoint found! Proceeding with initialized weights.")
        
    model.eval()

    # Setup hooks to extract layer outputs
    layer_outputs = {}
    def get_hook(name):
        def hook(module, input, output):
            x = output[0] if isinstance(output, tuple) else output
            layer_outputs[name] = x.detach()
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(get_hook(f"block_{i}")))

    # Prepare Image
    img_path = 'test_image.jpg'
    img_size = config['training'].get('image_size', 256)
    
    if os.path.exists(img_path):
        print(f"Loading image {img_path}")
        img = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(img).unsqueeze(0)
        img_tensor = TF.resize(img_tensor, [img_size, img_size], interpolation=TF.InterpolationMode.BILINEAR)
        img_tensor = img_tensor * 2.0 - 1.0  # Normalize to [-1, 1] range expected
    else:
        print(f"Image {img_path} not found. Using a random noise tensor as placeholder.")
        img_tensor = torch.randn(1, 3, img_size, img_size) * 0.5
        
    x0 = img_tensor.to(device)
    x0_noise = torch.randn_like(x0)
    
    # Unconditional Generation (using pad_idx)
    y = torch.full((1, txt_max_length), pad_idx, dtype=torch.long, device=device)

    # Simulation loop parameters
    steps = 11
    t_vals = torch.linspace(0.0, 1.0, steps=steps)
    
    all_mean_sims = {}
    num_blocks = len(model.blocks)
    
    # Iterate from t = 0 to 1 with 0.1 step
    for step_idx, t_val in enumerate(t_vals):
        t_val = t_val.item()
        
        # Noise the image
        xt = t_val * x0 + (1.0 - t_val) * x0_noise
        
        # Map time variable from [0, 1] to [0, 1000] scale
        t_cond = torch.tensor([t_val * 1000], device=device)
        
        layer_outputs.clear()
        with torch.no_grad():
            x0_pred = model(xt, t_cond, y)
            
        # Save generated/noised images into a folder
        mean = torch.tensor([0.6569382548332214, 0.5977839827537537, 0.5958537459373474], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.3143513798713684, 0.31483596563339233, 0.30866608023643494], device=device).view(1, 3, 1, 1)
        save_image(torch.clamp((xt * std) + mean, 0, 1), f"images/xt_t_{t_val:.1f}.png")
        save_image(torch.clamp((x0_pred * std) + mean, 0, 1), f"images/pred_t_{t_val:.1f}.png")
        
        print(f"\n--- Timestep {t_val:.1f} ---")
        
        # Save activations and extract image tokens
        img_tokens_per_layer = []
        for i in range(num_blocks):
            out_curr = layer_outputs[f"block_{i}"]
            # Isolate the image tokens (which start after txt_max_length)
            img_curr = out_curr[:, txt_max_length:]
            img_tokens_per_layer.append(img_curr)
            
            # Save raw activation tensor
            torch.save(img_curr.cpu(), f"activations/act_t_{t_val:.1f}_block_{i}.pt")
            
        # Calculate Cosine Sim Matrix
        sim_matrix = np.zeros((num_blocks, num_blocks))
        for i in range(num_blocks):
            for j in range(num_blocks):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    sim = F.cosine_similarity(img_tokens_per_layer[i], img_tokens_per_layer[j], dim=-1).mean().item()
                    sim_matrix[i, j] = sim
                    
        # Plotting Heatmap for this timestep
        plt.figure(figsize=(10, 8))
        # Plot heatmap
        plt.imshow(sim_matrix, cmap='Blues', vmin=0.0, vmax=1.0)
        plt.colorbar(label='Cosine Similarity')
        
        # Add labels and ticks
        plt.title(f"Cosine Similarity Heatmap (t={t_val:.1f})")
        plt.xlabel("Blocks")
        plt.ylabel("Blocks")
        ticks = np.arange(num_blocks)
        labels = [f"block_{i}" for i in range(num_blocks)]
        plt.xticks(ticks, labels, rotation=90)
        plt.yticks(ticks, labels)
        
        # Add a subtle grid
        plt.grid(True, which='minor', color='w', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        heatmap_path = f"images/heatmap_t_{t_val:.1f}.png"
        plt.savefig(heatmap_path)
        plt.close()
        
        print(f"Cosine similarity heatmap saved to {heatmap_path}.")

    # Cleanup hooks
    for h in handles:
        h.remove()

    print("\nDone!")
    print("Images and heatmaps saved to 'images/' and activations saved to 'activations/'")

if __name__ == "__main__":
    main()