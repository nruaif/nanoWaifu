import torch
import torch.nn as nn
from model import CoAtNeXtEncoder
import torch.nn.functional as F

def info_nce_loss(z1, z2, temperature=0.07):
    N = z1.size(0)
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temperature
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device),
    ])
    return F.cross_entropy(sim, labels)

def test_model():
    print("Testing CoAtNeXtEncoder Forward & Backward Pass...")
    
    # Initialize the model
    proj_dim = 8192
    model = CoAtNeXtEncoder(
        backbone_model='coatnext_nano_rw_224.sw_in1k',
        proj_dim=proj_dim,
        pretrained=False  # Faster for testing
    )
    
    # Create dummy images (B, C, H, W) -> [-1, 1] normalized
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    
    # Forward passes
    bin1, logits1 = model(x1)
    bin2, logits2 = model(x2)
    
    print(f"  Inputs: {x1.shape}")
    print(f"  Binary Shape: {bin1.shape}")
    print(f"  Logits Shape: {logits1.shape}")
    
    assert bin1.shape == (2, proj_dim)
    assert logits1.shape == (2, proj_dim)
    
    # Calculate losses
    nce_temperature = 0.07
    l1_weight = 1e-4
    
    infonce_loss = info_nce_loss(logits1, logits2, temperature=nce_temperature)
    l1_loss = l1_weight * torch.mean(torch.abs(bin1))
    
    total_loss = infonce_loss + l1_loss
    
    print(f"  Total Loss: {total_loss.item():.4f} | InfoNCE: {infonce_loss.item():.4f} | L1: {l1_loss.item():.4f}")
    
    # Backward pass
    total_loss.backward()
    
    # Verifying gradients
    assert model.proj[-1].weight.grad is not None, "Gradients missing on MLP projector"
    print("Forward and Backward pass completed successfully!")
    
if __name__ == "__main__":
    test_model()
    print("\nAll tests passed!")