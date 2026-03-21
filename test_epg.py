import torch
import torch.nn as nn
from model_epg import EPGEncoder, EPGDecoder, EPGModel
from torchvision.utils import save_image
import os

def test_epg():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")
    
    # Hyperparameters
    embed_dim = 256
    depth = 4
    num_heads = 4
    image_size = 64
    patch_size = 8
    num_classes = 100
    
    # 1. Initialize Components
    # Integrated EmbeddingBag inside EPGEncoder
    encoder = EPGEncoder(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, num_classes=num_classes).to(device)
    decoder = EPGDecoder(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads).to(device)
    model = EPGModel(encoder, decoder).to(device)
    
    # 2. Test Stage 1 Encoding
    print("Testing Stage 1 Encoding with integrated tags...")
    x = torch.randn(2, 3, image_size, image_size, device=device)
    t = torch.rand(2, device=device)
    t_scaled = 1000 * 0.25 * torch.log(t.clamp(min=1e-8))
    
    # Mock tags
    y_indices = torch.randint(0, num_classes, (10,), device=device)
    y_offsets = torch.tensor([0, 5], device=device)
    
    feat = model(x, t_scaled, y_indices=y_indices, y_offsets=y_offsets, stage=1)
    print(f"Stage 1 output shape: {feat.shape}")
    assert feat.shape == (2, embed_dim)
    
    # 3. Test Stage 2 Generation (Forward Pass)
    print("Testing Stage 2 Forward Pass...")
    out = model(x, t_scaled, y_indices=y_indices, y_offsets=y_offsets, stage=2)
    print(f"Stage 2 output shape: {out.shape}")
    assert out.shape == (2, 3, image_size, image_size)
    
    # 4. Test Generation (Sampling)
    print("Testing Flow Matching Sampling...")
    samples = model.sample_flow(image_size=image_size, batch_size=4, device=device, y_indices=y_indices[:4], y_offsets=torch.arange(4, device=device), steps=10)
    print(f"Sampled images shape: {samples.shape}")
    assert samples.shape == (4, 3, image_size, image_size)
    
    os.makedirs("test_outputs", exist_ok=True)
    save_image(samples, "test_outputs/sampled_test.png", nrow=2)
    print("Test passed!")

if __name__ == "__main__":
    test_epg()
