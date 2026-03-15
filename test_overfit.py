import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import random
from model_v2 import FCDMV2, TagProcessor, sample_flow
from torch.optim import AdamW
from torchvision.utils import save_image
import numpy as np

# Color Definitions
COLORS = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0]
}

# Position Definitions
POS_FNS = {
    "left": lambda x, y, size: 1.0 - (x / size),
    "right": lambda x, y, size: (x / size),
    "up": lambda x, y, size: 1.0 - (y / size),
    "down": lambda x, y, size: (y / size)
}

# Create all 24 fine-grained combined tags
COMBINED_TAGS = [f"{c}_{p}" for c in COLORS.keys() for p in POS_FNS.keys()]

def create_adaptive_gradient(tag, size=256):
    color_name, pos_name = tag.split('_')
    rgb = torch.tensor(COLORS[color_name]).view(3, 1, 1)
    pos_fn = POS_FNS[pos_name]
    
    grid_y, grid_x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    mask = pos_fn(grid_x.float(), grid_y.float(), size)
    
    image = rgb * mask
    return (image * 2 - 1) # Scale to [-1, 1]

def test_overfit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class MockTagProcessor:
        def __init__(self, tags):
            self.all_tags = tags
            self.tag_to_idx = {tag: i for i, tag in enumerate(self.all_tags)}
            self.num_classes = len(self.all_tags)
        
        def process_prompts(self, prompts, device, dropout_prob=0.0):
            indices = []
            offsets = [0]
            for p in prompts:
                if random.random() < dropout_prob:
                    indices.append(self.num_classes) # Null tag
                else:
                    tags = p.split()
                    count = 0
                    for t in tags:
                        if t in self.tag_to_idx:
                            indices.append(self.tag_to_idx[t])
                            count += 1
                    if count == 0:
                        indices.append(self.num_classes)
                offsets.append(len(indices))
            
            indices = torch.tensor(indices, dtype=torch.long, device=device)
            offsets = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
            return indices, offsets

    mock_tag_processor = MockTagProcessor(COMBINED_TAGS)

    # Instantiate V2 Model with forced parameters
    model = FCDMV2(
        in_channels=3,
        base_channels=256, # Increased to 256
        num_blocks=2,
        num_classes=mock_tag_processor.num_classes,
        patch_size=8,
        use_t_cond=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-4)
    
    print(f"Training on {len(COMBINED_TAGS)} tags with CFG (Dropout: 0.1)")
    
    model.train()
    for step in range(1001):
        optimizer.zero_grad()
        
        # Pick 1-2 random combined tags
        n_tags = random.randint(1, 2)
        selected_tags = random.sample(COMBINED_TAGS, n_tags)
        
        target_image = torch.zeros((3, 256, 256), device=device)
        for tag in selected_tags:
            grad = create_adaptive_gradient(tag, 256).to(device)
            target_image += grad
        
        target_image = torch.clamp(target_image, -1.0, 1.0)
        prompt = " ".join(selected_tags)
        
        t = torch.rand((1,), device=device)
        noise = torch.randn_like(target_image)
        xt = (1 - t) * target_image + t * noise
        
        # Training with CFG dropout
        y_indices, y_offsets = mock_tag_processor.process_prompts([prompt], device, dropout_prob=0.1)
        
        pred = model(xt.unsqueeze(0), t, y_indices, y_offsets)
        loss = F.mse_loss(pred, target_image.unsqueeze(0))
        
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}, Prompt: {prompt}")

    print("Sampling tests with CFG Scale 1.4...")
    model.eval()
    os.makedirs("test_results", exist_ok=True)
    
    test_prompts = [
        "red_left",
        "blue_right",
        "green_up",
        "magenta_down",
        "yellow_left blue_right"
    ]
    
    with torch.no_grad():
        for i, p in enumerate(test_prompts):
            # Save Ground Truth
            gt_image = torch.zeros((3, 256, 256), device=device)
            for tag in p.split():
                gt_image += create_adaptive_gradient(tag, 256).to(device)
            gt_image = torch.clamp(gt_image, -1.0, 1.0)
            save_image((gt_image / 2 + 0.5).clamp(0, 1), f"test_results/overfit_gt_{i}.png")

            # Sample with CFG
            samples = sample_flow(
                model, mock_tag_processor, 256, 1, [p], device, 
                steps=100, cfg_scale=1
            )
            save_image(samples, f"test_results/overfit_sample_{i}.png")
            print(f"Saved: test_results/overfit_sample_{i}.png (Prompt: {p})")

if __name__ == "__main__":
    test_overfit()
