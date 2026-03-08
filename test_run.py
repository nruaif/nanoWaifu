import torch
import torch.nn as nn
from model import CoAtNeXtEncoder, SIGReg

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
    lamb_lejepa = 0.02
    l1_weight = 1e-4
    
    inv_loss = (logits1 - logits2).square().mean()
    sigreg_loss = SIGReg(logits1, global_step=0, num_slices=256, chunk_size=32)
    l1_loss = l1_weight * torch.mean(torch.abs(bin1))
    
    total_loss = (lamb_lejepa * sigreg_loss) + ((1 - lamb_lejepa) * inv_loss) + l1_loss
    
    print(f"  Total Loss: {total_loss.item():.4f} | SIGReg: {sigreg_loss.item():.4f} | INV: {inv_loss.item():.4f} | L1: {l1_loss.item():.4f}")
    
    # Backward pass
    total_loss.backward()
    
    # Verifying gradients
    assert model.proj[-1].weight.grad is not None, "Gradients missing on MLP projector"
    print("Forward and Backward pass completed successfully!")
    
if __name__ == "__main__":
    test_model()
    print("\nAll tests passed!")