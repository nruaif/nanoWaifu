import torch
import yaml
from ar_transformer import ARTransformer
from vae import CategoricalVAE
import math

def test_generation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Setup minimal config or load from config_ar.yaml
    try:
        with open('config_ar.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {
            'model': {
                'latent_discrete': 256,
                'latent_continuous': 0,
                'dim': 512,
                'depth': 4, # Smaller for testing
                'heads': 8
            }
        }

    # 2. Initialize Model
    num_classes = 1000
    model = ARTransformer(
        num_classes=num_classes,
        latent_dim=config['model']['latent_discrete'],
        dim=config['model'].get('dim', 512),
        depth=config['model'].get('depth', 4),
        num_heads=config['model'].get('heads', 8),
        max_seq_len=512
    ).to(device)
    
    # Overwrite zero-init for testing
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param, std=0.1)
    
    model.eval().to(device)

    # 3. Dummy tags for conditioning
    # B=1, 5 tags
    class_indices = torch.randint(0, num_classes, (1, 5), device=device)
    
    print("🚀 Starting generation...")
    # Generate a small number of patches for quick test
    # 8x8 grid = 64 patches
    max_patches = 64
    
    try:
        with torch.no_grad():
            generated_latents = model.generate(
                class_indices, 
                max_patches=max_patches, 
                device=device
            )
        
        print(f"✅ Generation successful!")
        print(f"Generated latents shape: {generated_latents.shape}")
        print(f"Latents range: [{generated_latents.min().item()}, {generated_latents.max().item()}]")
        print(f"Unique values: {torch.unique(generated_latents)}")
        
        # Verify it can be reshaped for VAE
        grid_size = int(math.sqrt(max_patches))
        reshaped = generated_latents.view(1, grid_size, grid_size, config['model']['latent_discrete'])
        print(f"Reshaped for VAE: {reshaped.shape}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation()
