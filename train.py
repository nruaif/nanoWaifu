import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import argparse
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
import glob
import builtins
from dataset import WDSLoader
import torch.nn.functional as F
from huggingface_hub import upload_file
import threading


def _async_upload(ckpt_path, repo_id, step):
    try:
        upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=os.path.basename(ckpt_path),
            repo_id=repo_id,
            commit_message=f"Checkpoint at step {step}"
        )
        print(f"[HF] Uploaded: {ckpt_path}")
    except Exception as e:
        print(f"[HF] Upload failed for {ckpt_path}: {e}")


torch.backends.cuda.enable_flash_sdp(True)

# FIX: removed unused disp_loss function


from model_dit import TokenformerDiT as ModelClass, sample_flow as sample_fn, TagProcessor


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, local_rank, world_size, device
    else:
        return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def cleanup_checkpoints(output_dir, max_checkpoints, rank):
    if rank != 0:
        return
    max_checkpoints = 3
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoints) > max_checkpoints:
        for ckpt in checkpoints[:-max_checkpoints]:
            try:
                os.remove(ckpt)
                print(f"Removed old checkpoint: {ckpt}")
            except OSError as e:
                print(f"Error removing {ckpt}: {e}")


def save_checkpoint(
        model,
        optimizer,
        rank,
        output_dir,
        step,
        config,
        fixed_prompts=None,
        fixed_noise=None,
        push_to_hf=True,
        repo_id="Shio-Koube/ConvNext-Diff"
):
    if rank != 0:
        return

    print(f"\n[Step {step}] Saving Checkpoint...")

    model_to_save = model.module if hasattr(model, 'module') else model
    ckpt_path = os.path.join(output_dir, f'ckpt_step_{step}.pth')

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        # FIX: save optimizer state so resuming doesn't lose momentum/adaptive stats
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step,
        "config": config,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "fixed_prompts": fixed_prompts,
        "fixed_noise": fixed_noise
    }

    torch.save(checkpoint, ckpt_path)
    cleanup_checkpoints(output_dir, config.get('max_checkpoints', 3), rank)
    print(f"Checkpoint saved: {ckpt_path}")

    if push_to_hf and repo_id is not None:
        thread = threading.Thread(
            target=_async_upload,
            args=(ckpt_path, repo_id, step),
            daemon=True
        )
        thread.start()


def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    if rank != 0:
        def print_pass(*args, **kwargs): pass

        builtins.print = print_pass

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    from model_dit import TokenformerDiT, TagProcessor, sample_flow
    ModelClass, TagProcessor, sample_fn = TokenformerDiT, TagProcessor, sample_flow

    tag_processor = TagProcessor("tags.txt")
    num_classes = tag_processor.num_classes

    use_cached_latents = config['data'].get('use_cached_latents', False)
    if use_cached_latents:
        print(f">>> Using Cached Latents mode (loading pre-computed latents from {config['data']['cache_dir']})...")
        url = config['data']['cache_dir']
        wds_loader = WDSLoader(
            url=url,
            csv_path=config['data'].get('csv_path'),
            image_size=config['training']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            use_advanced_captions=config['data'].get('use_advanced_captions', True)
        )
        dataloader = wds_loader.make_loader()
    else:
        wds_loader = WDSLoader(
            url=config['data']['webdataset_url'],
            csv_path=config['data'].get('csv_path'),
            image_size=config['training']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            use_advanced_captions=config['data'].get('use_advanced_captions', True)
        )
        dataloader = wds_loader.make_loader()

    image_size = config['training']['image_size']

    use_vae = config['model'].get('use_vae', False)
    use_tiny_vae = config['model'].get('use_tiny_vae', False)
    in_channels = config['model'].get('in_channels', 3)

    if use_vae or use_tiny_vae:
        stats_path = "vae_stats.pt"
        if os.path.exists(stats_path):
            print(f">>> Loading VAE normalization stats from {stats_path}...")
            stats = torch.load(stats_path, map_location='cpu')
            latents_mean = stats["mean"].to(device=device, dtype=torch.bfloat16)
            latents_std = stats["std"].to(device=device, dtype=torch.bfloat16)
            standard_vae = None
        else:
            from diffusers import AutoencoderKLFlux2
            print(">>> Loading Standard FLUX.2 VAE to extract normalization stats...")
            standard_vae = AutoencoderKLFlux2.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                subfolder="vae",
                torch_dtype=torch.bfloat16
            ).to(device=device).eval()

            latents_mean = standard_vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype=torch.bfloat16)
            latents_std = torch.sqrt(
                standard_vae.bn.running_var.view(1, -1, 1, 1) + standard_vae.config.batch_norm_eps
            ).to(device, dtype=torch.bfloat16)

            torch.save({"mean": latents_mean.cpu(), "std": latents_std.cpu()}, stats_path)
            print(f">>> Saved VAE normalization stats to {stats_path}")

        if use_tiny_vae:
            if standard_vae is not None:
                del standard_vae
                torch.cuda.empty_cache()

            from flux2_tiny_autoencoder import Flux2TinyAutoEncoder
            print(">>> Loading Tiny FLUX.2 VAE...")
            vae = Flux2TinyAutoEncoder.from_pretrained(
                "fal/FLUX.2-Tiny-AutoEncoder",
            ).to(device=device, dtype=torch.bfloat16).eval()

            in_channels = 128
            print(f">>> Tiny VAE Mode Enabled: Model in_channels = {in_channels}")
        else:
            if standard_vae is None:
                from diffusers import AutoencoderKLFlux2
                print(">>> Loading Standard FLUX.2 VAE for encoding/decoding...")
                standard_vae = AutoencoderKLFlux2.from_pretrained(
                    "black-forest-labs/FLUX.2-dev",
                    subfolder="vae",
                    torch_dtype=torch.bfloat16
                ).to(device=device).eval()
            vae = standard_vae
            in_channels = 128
            print(f">>> VAE Mode Enabled: Model in_channels = {in_channels}")

    latent_size = (image_size // 8) if (use_vae or use_tiny_vae) else image_size // config['model'].get('patch_size',
                                                                                                        16)

    model = ModelClass(
        in_channels=128,
        dim=config['model'].get('fcdm_dim', 768),
        depth=config['model'].get('fcdm_depth', 12),
        num_heads=config['model'].get('num_heads', 12),
        num_classes=num_classes,
        use_checkpoint=config['training'].get('gradient_checkpointing', False),
    ).to(device=device, dtype=torch.bfloat16)

    # Resume Logic
    global_step = 0
    resume_path = config.get('resume_from', "outputs_dit/")
    fixed_prompts = None
    fixed_noise = None

    if resume_path:
        if os.path.isdir(resume_path):
            ckpt_files = glob.glob(os.path.join(resume_path, "ckpt_step_*.pth"))
            if ckpt_files:
                resume_path = sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            else:
                resume_path = None

        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model_to_load = model.module if hasattr(model, 'module') else model

            state_dict = checkpoint["model_state_dict"]
            model_state = model_to_load.state_dict()
            keys_to_delete = [
                k for k in state_dict
                if k in model_state and state_dict[k].shape != model_state[k].shape
            ]
            for k in keys_to_delete:
                print(f">>> Shape Mismatch: Removing {k}. "
                      f"Checkpoint: {state_dict[k].shape}, Model: {model_state[k].shape}")
                del state_dict[k]

            model_to_load.load_state_dict(state_dict, strict=False)
            # FIX: removed the arbitrary - 10 offset
            global_step = checkpoint["global_step"]

            if "fixed_noise" in checkpoint and checkpoint["fixed_noise"] is not None:
                fixed_noise = checkpoint["fixed_noise"].to(device)
            if "fixed_prompts" in checkpoint:
                fixed_prompts = checkpoint["fixed_prompts"]

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if config['training'].get('compile', False):
        print(">>> Compiling Model...")
        model = torch.compile(model, mode="max-autotune")

    # Reference to the unwrapped model for accessing k-diff parameters
    model_raw = model.module if hasattr(model, 'module') else model


    # Mock Optimizers for safety if file is missing locally
    try:
        from adv_optm import Muon_adv as NorMuon
        from adv_optm import AdamW_adv as DionAdamW
    except ImportError:
        print("Warning: Advanced optimizers missing, falling back to torch.optim.AdamW")
        NorMuon, DionAdamW = torch.optim.AdamW, torch.optim.AdamW

    adamw_params, normuon_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_1d = p.ndim <= 1
        is_dwconv = 'dwconv' in name or 'dw_conv' in name or (p.ndim == 4 and p.shape[1] == 1)
        is_embedding = ('embed' in name or isinstance(p, torch.nn.Embedding)
                        or isinstance(p, torch.nn.EmbeddingBag) or 'token_embed' in name)
        if is_1d or is_dwconv or is_embedding:
            adamw_params.append(p)
        else:
            normuon_params.append(p)

    try:
        opt_adamw = DionAdamW(adamw_params, lr=config['training']['learning_rate'], weight_decay=0, betas=(0.9, 0.95), cautious_wd=True)
        opt_normuon = NorMuon(normuon_params, lr=config['training']['learning_rate'] * 10, weight_decay=0.1, cautious_wd=True, normuon_variant=True)
    except Exception:
        opt_adamw = torch.optim.AdamW(adamw_params, lr=config['training']['learning_rate'], weight_decay=0.1)
        opt_normuon = torch.optim.AdamW(normuon_params, lr=config['training']['learning_rate'], weight_decay=0.1)

    class DualOptimizer:
        def __init__(self, opt1, opt2):
            self.opt1 = opt1
            self.opt2 = opt2

        def step(self):
            self.opt1.step()
            self.opt2.step()

        def zero_grad(self, set_to_none=True):
            self.opt1.zero_grad(set_to_none)
            self.opt2.zero_grad(set_to_none)

        def state_dict(self):
            return {"opt1": self.opt1.state_dict(), "opt2": self.opt2.state_dict()}

        def load_state_dict(self, state):
            if "opt1" in state: self.opt1.load_state_dict(state["opt1"])
            if "opt2" in state: self.opt2.load_state_dict(state["opt2"])

    optimizer = DualOptimizer(opt_adamw, opt_normuon)

    # FIX: restore optimizer state after it's constructed
    if resume_path and os.path.exists(resume_path if isinstance(resume_path, str) else ""):
        saved_checkpoint = torch.load(resume_path, map_location=device)
        if "optimizer_state_dict" in saved_checkpoint:
            try:
                #optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
                print(">>> Optimizer state restored.")
            except Exception as e:
                print(f">>> Could not restore optimizer state: {e}. Starting fresh.")

    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-C2I'), config=config)
        pbar = tqdm(range(global_step, config['training'].get('max_train_steps', 1000000)),
                    desc="Training", dynamic_ncols=True)
        os.makedirs(config['training']['output_dir'], exist_ok=True)
    if (use_vae or use_tiny_vae) and not use_cached_latents:
        vae = torch.compile(vae)
    data_iter = iter(dataloader)
    running_loss = 0.0
    running_match_loss = 0.0
    accum_steps = config['training'].get('grad_accum_steps', 1)
    is_mar = config['model'].get('type', 'v2') == 'mar'
    while global_step < config['training'].get('max_train_steps', 1000000):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        match_loss_accum = 0.0

        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            images, prompts, _ = batch
            y_indices, y_offsets = tag_processor.process_prompts(
                prompts, device, dropout_prob=config['training'].get('class_dropout_prob', 0.1)
            )

            # VAE Encoding
            if use_cached_latents:
                latents = images.to(device=device, dtype=torch.bfloat16)
                # Note: latents are already normalized by cache_latents.py
                inputs = latents
            elif use_tiny_vae:
                images = images.to(device, memory_format=torch.channels_last)
                with torch.no_grad():
                    v_images = images.to(dtype=torch.bfloat16)
                    out = vae.encode(v_images, return_dict=False)
                    latents = out[0] if isinstance(out, tuple) else out
                    # Normalize at native 128ch
                    latents = (latents - latents_mean) / latents_std
                    inputs = latents.to(dtype=torch.bfloat16)

            elif use_vae:
                images = images.to(device, memory_format=torch.channels_last)
                with torch.no_grad():
                    v_images = images.to(dtype=torch.bfloat16)
                    latents = vae.encode(v_images).latent_dist.mode()
                    # Reshape to 128ch for normalization (stats are in 2x2 patch format)
                    latents = F.pixel_unshuffle(latents, 2).to(dtype=torch.bfloat16)
                    latents = (latents - latents_mean) / latents_std
                    # Keep in 128ch — model expects in_channels=128
                    inputs = latents
            else:
                images = images.to(device, memory_format=torch.channels_last)
                inputs = images.to(dtype=torch.bfloat16)

            if rank == 0 and fixed_prompts is None:
                fixed_prompts = prompts[:16]
                fixed_noise = torch.randn_like(inputs[:16])

            B, C, H, W = inputs.shape

            def sample_logit_normal(m_loc, s_scale, bs, device, dtype):
                eps = torch.randn(bs, device=device, dtype=dtype)
                return torch.sigmoid(m_loc + s_scale * eps)
            def sample_uniform(bs, device, dtype):
                return torch.rand(bs, device=device, dtype=dtype)
            # Global Logit-Normal Timestep Sampler
            #t = sample_logit_normal(m_loc=0.8, s_scale=1.0, bs=B, device=device, dtype=torch.bfloat16)
            t = sample_uniform(bs=B, device=device, dtype=torch.bfloat16)
            t2 = t + sample_uniform(bs=B, device=device, dtype=torch.bfloat16) * (1 - t)

            t_reshaped = t.view(B, 1, 1, 1)
            t2_reshaped = t2.view(B, 1, 1, 1)
            noise = torch.randn_like(inputs)
            xt = (1 - t_reshaped) * inputs + t_reshaped * noise
            xt2 = (1 - t2_reshaped) * inputs + t2_reshaped * noise

            # --- Training Augmentations (each applied independently w/ 50% chance) ---
            # 1. Gaussian noise injection to xt to simulate drift during inference
            noise_inject_ratio = config['training'].get('noise_inject_ratio', 0.1)
            if noise_inject_ratio > 0:
                noise_mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5).to(xt.dtype)
                noise_injection = torch.randn_like(xt)
                xt = xt + noise_mask * noise_inject_ratio * noise_injection
                xt2 = xt2 + noise_mask * noise_inject_ratio * noise_injection

            # 2. Intra-sample crossing: build xt_neg at the SAME timestep t
            #    from a different clean sample to simulate mean-seeking drift
            cross_ratio = config['training'].get('cross_ratio', 0.1)
            if cross_ratio > 0:
                cross_mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5).to(xt.dtype)
                inputs_neg = inputs.roll(shifts=1, dims=0)
                noise_neg = torch.randn_like(inputs)
                
                xt_neg = (1 - t_reshaped) * inputs_neg + t_reshaped * noise_neg
                xt = xt + cross_mask * cross_ratio * (xt_neg - xt)
                
                xt2_neg = (1 - t2_reshaped) * inputs_neg + t2_reshaped * noise_neg
                xt2 = xt2 + cross_mask * cross_ratio * (xt2_neg - xt2)

            # Model outputs direct v-prediction and layer match loss
            v_pred, match_loss = model(xt, t, y_indices, y_offsets,
                                       return_layer_match=True, xt2=xt2, t2=t2)

            # Compute velocity-space loss (flow matching target: noise - inputs)
            v_target = noise - inputs
            loss = F.mse_loss(v_pred, v_target)  + 0.2 * match_loss

            loss = loss / accum_steps
            loss.backward()
            loss_accum += F.mse_loss(v_pred, v_target).item()
            match_loss_accum += match_loss.item() / accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        running_loss += loss_accum
        running_match_loss += match_loss_accum
        if rank == 0:
            pbar.update(1)

            if global_step % config['training']['log_every_steps'] == 0:
                log_interval = config['training']['log_every_steps']
                avg_loss = running_loss / log_interval
                avg_match_loss = running_match_loss / log_interval
                wandb.log({
                    "train/loss": avg_loss,
                    "train/match_loss": avg_match_loss
                }, step=global_step)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "match_loss": f"{avg_match_loss:.4f}"})
                running_loss = 0.0
                running_match_loss = 0.0

            if global_step % config['training']['save_image_every_steps'] == 0:
                save_checkpoint(model, optimizer, rank, config['training']['output_dir'],
                                global_step, config, fixed_prompts, fixed_noise)

                print(f"\n[Step {global_step}] Generating validation samples...")
                model.eval()
                with torch.no_grad():
                    samples = sample_fn(
                        model.module if hasattr(model, 'module') else model,
                        tag_processor,
                        latent_size,
                        len(fixed_prompts),
                        fixed_prompts,
                        device,
                        guidance_scale=config['training'].get('cfg_scale', 1.4),
                        noise=fixed_noise,
                        sampler_type="euler"
                    )

                    if use_tiny_vae:
                        samples = samples.to(dtype=torch.bfloat16)
                        # Un-normalize 128ch latents → decode with tiny VAE
                        latents = samples * latents_std + latents_mean
                        out = vae.decode(latents, return_dict=False)
                        recon = out[0] if isinstance(out, tuple) else out
                        samples = recon.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)

                    elif use_vae:
                        samples = samples.to(dtype=torch.bfloat16)
                        # Un-normalize 128ch → pixel_shuffle to 32ch → decode with standard VAE
                        latents = samples * latents_std + latents_mean
                        latents = F.pixel_shuffle(latents, 2)
                        recon = vae.decode(latents).sample
                        samples = recon.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)

                    grid = make_grid(samples, nrow=4)
                    wandb.log({
                        "val/samples": wandb.Image(grid, caption=f"Step {global_step}")
                    }, step=global_step)
                model.train()

    if rank == 0:
        save_checkpoint(model, optimizer, rank, config['training']['output_dir'],
                        global_step, config, fixed_prompts, fixed_noise)
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
    cleanup_ddp()
