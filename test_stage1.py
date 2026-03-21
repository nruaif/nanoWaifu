import torch
import torch.nn as nn
from model_epg import EPGEncoder, EPGProjector
from siglip import SupConLoss
import matplotlib.pyplot as plt
import os
import math
import numpy as np

def log_confusion_matrix_local(sim_matrix, path, batch_size):
    """Saves the mean Positive vs Negative similarity matrix locally."""
    k = sim_matrix.shape[0] // batch_size
    reshaped = sim_matrix.view(k, batch_size, k, batch_size)
    
    pos_sim = np.zeros((k, k))
    neg_sim = np.zeros((k, k))
    
    for i in range(k):
        for j in range(k):
            block = reshaped[i, :, j, :]
            pos_sim[i, j] = block.diagonal().mean().item()
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=block.device)
            neg_sim[i, j] = block[mask].mean().item()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pos_sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    views = ["Aug", "Noise", "Label"]
    ax.set_xticks(range(k)); ax.set_xticklabels(views)
    ax.set_yticks(range(k)); ax.set_yticklabels(views)
    
    for i in range(k):
        for j in range(k):
            color = "w" if pos_sim[i, j] < 0.5 else "k"
            ax.text(j, i, f"Pos: {pos_sim[i, j]:.2f}\nNeg: {neg_sim[i, j]:.2f}", 
                    ha="center", va="center", color=color, fontsize=9, fontweight='bold')
            
    ax.set_title("Group Alignment Contrast: Pos vs Neg (Test)")
    plt.savefig(path)
    plt.close(fig)
    print(f"Contrast similarity matrix saved to {path}")

def test_stage1_logic(image_size=64, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Testing Stage 1 Logic (Size: {image_size}x{image_size}, Batch: {batch_size}) ---")
    
    embed_dim, proj_dim, patch_size, num_classes = 128, 64, 8, 50
    encoder = EPGEncoder(patch_size=patch_size, embed_dim=embed_dim, depth=2, num_heads=4, num_classes=num_classes).to(device)
    projector = EPGProjector(embed_dim=embed_dim, proj_dim=proj_dim).to(device)
    supcon = SupConLoss().to(device)
    
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    t = torch.rand(batch_size, device=device)
    t_scaled = 1000 * 0.25 * torch.log(t.clamp(min=1e-8))
    y_indices = torch.randint(0, num_classes, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)
    
    # Forward passes
    y_feat = projector(encoder.get_y_feat(y_indices, y_offsets))
    q_im = projector(encoder(x, t_scaled, y_indices, y_offsets)[:, 0])
    q_noise = projector(encoder(x + 0.5 * torch.randn_like(x), t_scaled, y_indices, y_offsets)[:, 0])
    
    loss, _, sim_matrix = supcon([q_im, q_noise, y_feat], [q_im, q_noise, y_feat])
    
    os.makedirs("test_outputs", exist_ok=True)
    log_confusion_matrix_local(sim_matrix, f"test_outputs/alignment_contrast_{image_size}.png", batch_size)

if __name__ == "__main__":
    test_stage1_logic(image_size=64)
    print("\nAll Stage 1 tests passed!")
