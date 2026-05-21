import torch
import torch.nn as nn
from model_dit import TokenformerDiT
import glob
import os
import yaml
import numpy as np

def log_tensor_stats(t, name, stage):
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return
        
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    max_val = t.abs().max().item() if t.numel() > 0 else 0
    
    sus = False
    reasons = []
    if has_nan:
        sus = True
        reasons.append("NaN")
    if has_inf:
        sus = True
        reasons.append("Inf")
    if max_val > 100.0:  # Lowered threshold to 100.0 for finer grain
        sus = True
        reasons.append(f"Large Max ({max_val:.2f})")
        
    if sus:
        # Find index of max value for finer grain debugging
        flat_idx = torch.argmax(t.abs()).item()
        idx = np.unravel_index(flat_idx, t.shape) if t.numel() > 0 else None
        
        mean_val = t.mean().item()
        std_val = t.std().item()
        
        print(f"[SUS ACT] {stage}: {name} | Reason(s): {', '.join(reasons)}")
        print(f"          Shape: {t.shape} | Mean: {mean_val:.4f} | Std: {std_val:.4f}")
        print(f"          Max value found at index {idx} -> {t.flatten()[flat_idx].item():.4f}")

def check_sus_act_pre(module, args, name):
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            log_tensor_stats(arg, f"{name} (Input arg {i})", "PRE")
        elif isinstance(arg, tuple):
            for j, a in enumerate(arg):
                log_tensor_stats(a, f"{name} (Input arg {i}.{j})", "PRE")

def check_sus_act_post(module, input, output, name):
    if isinstance(output, torch.Tensor):
        outputs = [output]
    elif isinstance(output, tuple):
        outputs = [o for o in output if isinstance(o, torch.Tensor)]
    else:
        return
        
    for i, out in enumerate(outputs):
        log_tensor_stats(out, f"{name} (Output {i})", "POST")

def get_newest_checkpoint(output_dir):
    ckpt_files = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    if not ckpt_files:
        return None
    return sorted(ckpt_files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))[-1]

def main():
    print("Loading config...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    use_vae = config['model'].get('use_vae', False)
    use_tiny_vae = config['model'].get('use_tiny_vae', False)
    in_channels = config['model'].get('in_channels', 3)
    if use_vae or use_tiny_vae:
        in_channels = 128
        
    print("Initializing model...")
    device = torch.device("cpu")
    model = TokenformerDiT(
        in_channels=in_channels,
        dim=config['model'].get('fcdm_dim', 768),
        depth=config['model'].get('fcdm_depth', 12),
        num_heads=config['model'].get('num_heads', 12),
        num_classes=12476, # Default from TokenformerDiT definition
    ).to(device)
    
    output_dir = config['training'].get('output_dir', 'outputs_dit')
    newest_ckpt = get_newest_checkpoint(output_dir)
    
    if newest_ckpt:
        print(f"Found newest checkpoint: {newest_ckpt}")
        checkpoint = torch.load(newest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found in {output_dir}, using randomly initialized weights.")
    
    # Register hooks
    for name, module in model.named_modules():
        module.register_forward_pre_hook(lambda m, args, n=name: check_sus_act_pre(m, args, n))
        module.register_forward_hook(lambda m, i, o, n=name: check_sus_act_post(m, i, o, n))
        
        
    print("Creating dummy inputs...")
    batch_size = 2
    H, W = 8, 8
    
    x_in = torch.randn(batch_size, in_channels, H, W, device=device)

    
    y_indices = torch.randint(0, 100, (batch_size * 3,), device=device)
    y_offsets = torch.arange(0, batch_size * 3, 3, device=device)
    t_global = torch.rand(batch_size, device=device)
    
    print("Running forward pass...")
    try:
        model(x_in, t_global, y_indices, y_offsets)
        print("Forward pass completed.")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == '__main__':
    main()
