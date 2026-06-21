import torch
import sys
import os

# Add diffusers directory to path so we can import mmdit
sys.path.append(os.path.join(os.path.dirname(__file__), 'minit2i-torch', 'diffusers'))

from mmdit import DiffusionModel, MMJiTConfig
from model_dit import MiniT2IWrapper

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Init small model
    print("Initializing small MMJiT config...")
    cfg = MMJiTConfig(
        image_size=64,
        patch_size=16,
        in_channels=3,
        txt_input_size=256,
        hidden_size=128,
        txt_hidden_size=128,
        cond_vec_size=128,
        depth_double=2,
        txt_preamble_depth=1,
        num_heads=4,
        head_dim=32,
        pca_channels=32,
    )
    transformer = DiffusionModel(cfg)
    
    num_classes = 100
    seq_len = 32
    print("Wrapping model in MiniT2IWrapper...")
    wrapper = MiniT2IWrapper(num_classes=num_classes, seq_len=seq_len, transformer=transformer)
    wrapper.to(device)
    
    # Dummy data
    B = 2
    images = torch.randn(B, 3, 64, 64, device=device)
    t = torch.rand(B, device=device)
    
    # Dummy tags: 3 tags for first image, 2 tags for second
    y_indices = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long, device=device)
    y_offsets = torch.tensor([0, 3], dtype=torch.long, device=device)
    
    print("Running forward pass (train mode)...")
    wrapper.train()
    v_pred, match_loss = wrapper(images, t, y_indices, y_offsets, return_layer_match=True)
    print(f"Forward pass successful. Output shape: {v_pred.shape}")
    
    print("Running backward pass...")
    loss = v_pred.mean()
    loss.backward()
    print("Backward pass successful. Gradients computed.")
    
    print("Running generation (sample)...")
    wrapper.eval()
    with torch.no_grad():
        samples = wrapper.sample(y_indices, y_offsets, image_size=64, num_inference_steps=2)
    print(f"Sample generation successful. Output shape: {samples.shape}")
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test()
