import torch
import torch.nn.functional as F
from model_v2 import FCDMV2, TagProcessor, sample_flow

def test_overfit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mock tags
    with open("test_tags.txt", "w") as f:
        f.write("1girl\nblue_hair\nsolo\n")
    
    tag_processor = TagProcessor("test_tags.txt")
    model = FCDMV2(
        in_channels=3,
        base_channels=32,
        num_blocks=1,
        num_classes=tag_processor.num_classes,
        patch_size=8
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Single image overfit
    image = torch.randn(1, 3, 64, 64).to(device)
    prompts = ["1girl blue_hair solo"]
    y_indices, y_offsets = tag_processor.process_prompts(prompts, device)

    print("Starting overfit test...")
    for i in range(200):
        t = torch.rand((1,), device=device)
        noise = torch.randn_like(image)
        
        # Current (incorrect) training logic in train.py:
        # xt = (1 - t) * image + t * noise
        # pred = model(xt, t, y_indices, y_offsets)
        # loss = F.mse_loss(pred, image)
        
        # Rectified Flow (Targeting velocity v = noise - image)
        t_reshaped = t.view(-1, 1, 1, 1)
        xt = (1 - t_reshaped) * image + t_reshaped * noise
        target = noise - image
        
        pred = model(xt, t, y_indices, y_offsets)
        loss = F.mse_loss(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Step {i}, Loss: {loss.item():.6f}")

    print("Sampling...")
    samples = sample_flow(model, tag_processor, 64, 1, prompts, device, steps=20)
    print("Sample shape:", samples.shape)
    # Check if samples are somewhat close to original image (highly informal check)
    # samples are in [0, 1], image is randn. Let's just see if it runs.
    print("Test passed (it runs).")

if __name__ == "__main__":
    test_overfit()
