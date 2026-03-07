import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from model import TREADDiT, ImageTagger


@torch.no_grad()
def sample_flow_test(model, batch_size, tag_vecs, coords, device, steps=50, cfg_scale=4.0):
    """Standalone sample_flow for testing (no train.py dependency)."""
    model_engine = model.module if isinstance(model, DDP) else model
    latent_ch = model_engine.backbone.in_channels
    latent_sz = model_engine.backbone.input_size
    x = torch.randn((batch_size, latent_ch, latent_sz, latent_sz), device=device)

    dt = 1.0 / steps
    indices = torch.linspace(0, 1, steps, device=device)
    null_tags = torch.zeros_like(tag_vecs)

    for i in range(steps):
        t = indices[i]
        x_in = torch.cat([x, x])
        t_batch = torch.full((batch_size * 2,), t.item(), device=device, dtype=torch.float)
        tags_in = torch.cat([tag_vecs, null_tags])
        coords_in = torch.cat([coords, coords])

        # No image_tags during sampling (only text tags + CFG)
        v_pred, _ = model(x_in, t_batch * 1000, tags_in, coords_in, image_tags=None, drop_rate=0.0)
        v_cond, v_uncond = v_pred.chunk(2)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + v * dt

    return x


def test_model():
    print("Testing TREADDiT + ImageTagger Forward Pass...")
    num_tags = 20
    num_image_tags = 64  # Small for test

    model = TREADDiT(
        input_size=16,
        patch_size=1,
        in_channels=128,
        hidden_size=64,
        depth=6,
        num_heads=2,
        num_tags=num_tags,
        class_dropout_prob=0.1,
        routing_start=1,
        routing_end=5,
        num_image_tags=num_image_tags,
    )

    tagger = ImageTagger(num_binary_channels=num_image_tags, pretrained=False)

    # Simulate latent input and raw image
    x_latent = torch.randn(2, 128, 16, 16)
    x_raw = torch.randn(2, 3, 256, 256)  # Raw image for tagger
    t = torch.rand((2,)) * 1000
    tags = (torch.rand(2, num_tags) < 0.3).float()
    coords = torch.rand(2, 4)

    # Get image tags
    image_tags = tagger(x_raw)
    print(f"  Image tags shape: {image_tags.shape}, unique values: {image_tags.unique().tolist()[:5]}...")
    assert image_tags.shape == (2, num_image_tags)

    # Test without image tags
    out, x_bb = model(x_latent, t, tags, coords, image_tags=None, drop_rate=0.0)
    print(f"  No image_tags | Head: {out.shape}, Backbone: {x_bb.shape}")
    assert out.shape == x_latent.shape

    # Test with image tags + routing
    out_r, x_bb_r = model(x_latent, t, tags, coords, image_tags=image_tags, drop_rate=0.5)
    print(f"  With image_tags + routing | Head: {out_r.shape}, Backbone: {x_bb_r.shape}")
    assert out_r.shape == x_latent.shape

    print("Forward pass successful.")

    print("Testing Backward Pass (joint tagger + model)...")
    loss = out_r.mean() + x_bb_r.mean()
    loss.backward()
    # Verify tagger gradients flow
    assert tagger.proj.weight.grad is not None, "Tagger gradients missing!"
    print("Backward pass successful - tagger receives gradients.")


def test_flow_matching():
    print("Testing Flow Matching with ImageTagger...")
    num_tags = 10
    num_image_tags = 32

    model = TREADDiT(
        input_size=16,
        patch_size=1,
        in_channels=128,
        hidden_size=32,
        depth=6,
        num_heads=1,
        num_tags=num_tags,
        routing_start=1,
        routing_end=5,
        num_image_tags=num_image_tags,
    )

    tagger = ImageTagger(num_binary_channels=num_image_tags, pretrained=False)

    x1 = torch.randn(2, 128, 16, 16)
    x_raw = torch.randn(2, 3, 64, 64)
    tags = (torch.rand(2, num_tags) < 0.3).float()
    coords = torch.rand(2, 4)

    image_tags = tagger(x_raw)

    t = torch.rand((2,))
    x0 = torch.randn_like(x1)
    t_reshaped = t.view(-1, 1, 1, 1)
    xt = (1 - t_reshaped) * x0 + t_reshaped * x1
    ut = x1 - x0

    vt, x_bb = model(xt, t * 1000, tags, coords, image_tags=image_tags, drop_rate=0.5)
    loss_head = torch.mean((vt - ut) ** 2)
    loss_bb = torch.mean((x_bb - x1) ** 2)
    loss_l1 = 1e-4 * torch.mean(torch.abs(image_tags))
    loss = loss_head + loss_bb + loss_l1
    print(f"  Loss (Total): {loss.item():.4f} | L1: {loss_l1.item():.4f}")

    # Sampling (no image tags, no routing)
    samples = sample_flow_test(model, 1, tags[:1], coords[:1], "cpu", steps=5)
    print(f"  Sample shape: {samples.shape}")
    assert samples.shape == (1, 128, 16, 16)
    print("Flow matching test passed.")


if __name__ == "__main__":
    test_model()
    test_flow_matching()
    print("\nAll tests passed!")