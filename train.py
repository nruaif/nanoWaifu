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
import torch.nn.functional as F
from huggingface_hub import upload_file
import threading

from model_dit import PixNerDiT as ModelClass

class TagProcessor:
    def __init__(self, tags_file, max_tags=32):
        with open(tags_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.pad_idx = len(self.tags)
        self.num_classes = len(self.tags) + 1
        self.max_tags = max_tags

    def process_prompts(self, prompts, device):
        batch_indices = []
        for p in prompts:
            tags = p.split()
            indices = []
            for t in tags:
                if t in self.tag_to_idx:
                    indices.append(self.tag_to_idx[t])
            indices = indices[:self.max_tags]
            indices += [self.pad_idx] * (self.max_tags - len(indices))
            batch_indices.append(indices)
        return torch.tensor(batch_indices, dtype=torch.long, device=device)


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

    tag_processor = TagProcessor("tags.txt")
    num_classes = tag_processor.num_classes

    # Mock WDSLoader for runtime safety if missing locally
    try:
        from dataset import WDSLoader
        wds_loader = WDSLoader(
            url=config['data']['webdataset_url'],
            csv_path=config['data'].get('csv_path'),
            image_size=config['training']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            use_advanced_captions=config['data'].get('use_advanced_captions', True)
        )
        dataloader = wds_loader.make_loader()
    except ImportError:
        print("Warning: Mock dataset loaded.")

        class MockLoader:
            def __iter__(self): return self

            def __next__(self):
                return torch.randn(config['training']['batch_size'], 3, config['training']['image_size'],
                                   config['training']['image_size']), ["test"] * config['training']['batch_size'], None

        dataloader = MockLoader()

    image_size = config['training']['image_size']

    use_vae = config['model'].get('use_vae', False)
    use_tiny_vae = config['model'].get('use_tiny_vae', False)
    in_channels = config['model'].get('in_channels', 3)

    if use_vae or use_tiny_vae:
        from diffusers import AutoencoderKLFlux2
        print(">>> Loading Standard FLUX.2 VAE to extract normalization stats...")
        standard_vae = AutoencoderKLFlux2.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to(device=device).eval()

        latents_mean = standard_vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        latents_std = torch.sqrt(
            standard_vae.bn.running_var.view(1, -1, 1, 1) + standard_vae.config.batch_norm_eps
        ).to(device)

        if use_tiny_vae:
            del standard_vae
            torch.cuda.empty_cache()
            from flux2_tiny_autoencoder import Flux2TinyAutoEncoder
            print(">>> Loading Tiny FLUX.2 VAE...")
            vae = Flux2TinyAutoEncoder.from_pretrained(
                "fal/FLUX.2-Tiny-AutoEncoder",
            ).to(device=device, dtype=torch.bfloat16).eval()
            in_channels = 128
            print(f">>> Tiny VAE Mode Enabled: Model in_channels adjusted to {in_channels}")
        else:
            vae = standard_vae
            in_channels = 128
            print(f">>> VAE Mode Enabled: Model in_channels adjusted to {in_channels}")

    patch_size = config['model'].get('patch_size', 2)

    model = ModelClass(
        in_channels=in_channels,
        hidden_size=config['model'].get('hidden_size', 1152),
        num_groups=config['model'].get('num_heads', 12),
        patch_size=patch_size,
        txt_embed_dim=num_classes,
        txt_max_length=tag_processor.max_tags,
    ).to(device=device)

    if config['training'].get('gradient_checkpointing', False):
        model.enable_gradient_checkpointing()
        print(">>> Gradient Checkpointing Enabled")

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
                print(f">>> Shape Mismatch: Removing {k}")
                del state_dict[k]

            model_to_load.load_state_dict(state_dict, strict=False)
            global_step = checkpoint["global_step"]

            if "fixed_noise" in checkpoint and checkpoint["fixed_noise"] is not None:
                fixed_noise = checkpoint["fixed_noise"].to(device)
            if "fixed_prompts" in checkpoint:
                fixed_prompts = checkpoint["fixed_prompts"]

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if config['training'].get('compile', False):
        print(">>> Compiling Model...")
        model.compile()

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
        opt_adamw = DionAdamW(adamw_params, lr=config['training']['learning_rate'], weight_decay=0.1, betas=(0.9, 0.95))
        opt_normuon = NorMuon(normuon_params, lr=config['training']['learning_rate'], weight_decay=0.1)
    except Exception:
        opt_adamw = torch.optim.AdamW(adamw_params, lr=config['training']['learning_rate'], weight_decay=0.1)
        opt_normuon = torch.optim.AdamW(normuon_params, lr=config['training']['learning_rate'], weight_decay=0.1)

    class DualOptimizer:
        def __init__(self, opt1, opt2):
            self.opt1 = opt1
            self.opt2 = opt2

        def step(self):
            self.opt1.step(); self.opt2.step()

        def zero_grad(self, set_to_none=True):
            self.opt1.zero_grad(set_to_none); self.opt2.zero_grad(set_to_none)

        def state_dict(self):
            return {"opt1": self.opt1.state_dict(), "opt2": self.opt2.state_dict()}

        def load_state_dict(self, state):
            if "opt1" in state: self.opt1.load_state_dict(state["opt1"])
            if "opt2" in state: self.opt2.load_state_dict(state["opt2"])

    optimizer = DualOptimizer(opt_adamw, opt_normuon)

    if resume_path and os.path.exists(resume_path if isinstance(resume_path, str) else ""):
        saved_checkpoint = torch.load(resume_path, map_location=device)
        if "optimizer_state_dict" in saved_checkpoint:
            try:
                optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
                print(">>> Optimizer state restored.")
            except Exception as e:
                print(f">>> Could not restore optimizer state: {e}. Starting fresh.")

    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-C2I'), config=config)
        pbar = tqdm(range(global_step, config['training'].get('max_train_steps', 1000000)),
                    desc="Training", dynamic_ncols=True)
        os.makedirs(config['training']['output_dir'], exist_ok=True)

    data_iter = iter(dataloader)
    running_loss = 0.0
    accum_steps = config['training'].get('grad_accum_steps', 1)

    while global_step < config['training'].get('max_train_steps', 1000000):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0

        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            images, prompts, _ = batch
            if hasattr(images, "to"):
                images = images.to(device, memory_format=torch.channels_last)
            y_indices = tag_processor.process_prompts(prompts, device)

            # VAE Encoding
            if use_tiny_vae:
                with torch.no_grad():
                    v_images = images.to(dtype=torch.bfloat16)
                    out = vae.encode(v_images, return_dict=False)
                    latents = out[0] if isinstance(out, tuple) else out
                    latents = (latents - latents_mean) / latents_std
                    inputs = latents.to(dtype=torch.float32)
            elif use_vae:
                with torch.no_grad():
                    v_images = images.to(dtype=torch.bfloat16)
                    latents = vae.encode(v_images).latent_dist.mode()
                    latents = F.pixel_unshuffle(latents, 2)
                    inputs = ((latents - latents_mean) / latents_std).to(dtype=torch.float32)
            else:
                inputs = images.to(dtype=torch.float32)

            if rank == 0 and fixed_prompts is None:
                fixed_prompts = prompts[:16]
                fixed_noise = torch.randn_like(inputs[:16])

            B, C, H, W = inputs.shape

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Generate timestep
                t = torch.empty((B,), device=device).uniform_(0, 1.0)
                t_condition = t * 1000

                x0_noise = torch.randn_like(inputs)

                t_expand = t.view(B, 1, 1, 1)
                xt = t_expand * inputs + (1 - t_expand) * x0_noise

                # CFG Dropout Mask
                drop_prob = config['training'].get('class_dropout_prob', 0.1)
                drop_mask = torch.rand(B, device=device) < drop_prob
                
                y_input = y_indices.clone()
                y_input[drop_mask] = tag_processor.pad_idx

                x0_pred = model(xt, t_condition, y_input)

                # calculate v-space target
                target_v = inputs - x0_noise

                # convert x0_pred to v_pred with clipped t calculation
                t_clipped = torch.clamp(t_expand, max=0.95)
                v_pred = (x0_pred - xt) / (1.0 - t_clipped)

                mse_loss_raw = F.mse_loss(v_pred, target_v, reduction='none')
                mse_loss_mean = mse_loss_raw.mean()

                loss = mse_loss_mean / accum_steps
                
            loss.backward()
            loss_accum += mse_loss_mean.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        running_loss += loss_accum

        if rank == 0:
            pbar.update(1)
            if global_step % config['training']['log_every_steps'] == 0:
                log_interval = config['training']['log_every_steps']
                avg_loss = running_loss / log_interval
                log_dict = {"train/loss": avg_loss}

                wandb.log(log_dict, step=global_step)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                running_loss = 0.0

            if global_step % config['training']['save_image_every_steps'] == 0:
                save_checkpoint(model, optimizer, rank, config['training']['output_dir'],
                                global_step, config, fixed_prompts, fixed_noise)

                model.eval()
                with torch.no_grad():
                    print(f"\n[Step {global_step}] Generating validation samples...")
                    base_model = model.module if hasattr(model, 'module') else model
                    val_y = tag_processor.process_prompts(fixed_prompts, device)
                    
                    H_val, W_val = inputs.shape[2], inputs.shape[3]

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        samples = base_model.sample(
                            B=val_y.shape[0], H=H_val, W=W_val,
                            device=device, steps=50, y=val_y, cfg_scale=4.0, pad_idx=tag_processor.pad_idx
                        )

                    if use_tiny_vae or use_vae:
                        samples = samples.to(dtype=torch.bfloat16)
                        samples = (samples * latents_std) + latents_mean
                        if not use_tiny_vae:
                            samples = F.pixel_shuffle(samples, 2)
                            recon = vae.decode(samples).sample
                        else:
                            out = vae.decode(samples, return_dict=False)
                            recon = out[0] if isinstance(out, tuple) else out

                        samples = recon.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)
                    else:
                        samples = samples.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)

                    grid = make_grid(samples, nrow=4)
                    wandb.log({f"val/samples": wandb.Image(grid, caption=f"Validation @ Step {global_step}")}, step=global_step)
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
