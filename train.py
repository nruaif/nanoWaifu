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

# Import the new MiniT2I Wrapper
from model_dit import MiniT2IWrapper as ModelClass, TagProcessor
from dataset import WDSLoader

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

    print(">>> Initializing WDSLoader (Raw RGB mode)...")
    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'] if not config['data'].get('use_cached_latents', False) else config['data']['cache_dir'],
        csv_path=config['data'].get('csv_path'),
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        use_advanced_captions=config['data'].get('use_advanced_captions', True)
    )
    dataloader = wds_loader.make_loader()

    # Load MiniT2I Wrapper
    model = ModelClass(
        num_classes=num_classes,
        model_id="MiniT2I/MiniT2I",
        seq_len=32
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
                print(f">>> Shape Mismatch: Removing {k}. ")
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
        model = torch.compile(model, mode="max-autotune")

    model_raw = model.module if hasattr(model, 'module') else model

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

    if resume_path and os.path.exists(resume_path if isinstance(resume_path, str) else ""):
        saved_checkpoint = torch.load(resume_path, map_location=device)
        if "optimizer_state_dict" in saved_checkpoint:
            try:
                # optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])
                print(">>> Optimizer state restored.")
            except Exception as e:
                print(f">>> Could not restore optimizer state: {e}. Starting fresh.")

    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-MiniT2I'), config=config)
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
            
            y_indices, y_offsets = tag_processor.process_prompts(
                prompts, device, dropout_prob=config['training'].get('class_dropout_prob', 0.1)
            )

            # Ensure inputs are memory_format channels_last and bfloat16
            inputs = images.to(device, memory_format=torch.channels_last).to(dtype=torch.bfloat16)

            B, C, H, W = inputs.shape

            if rank == 0 and fixed_prompts is None:
                fixed_prompts = prompts[:16]
                fixed_noise = torch.randn_like(inputs[:16])

            # MiniT2I Log-Normal Timestep Sampler
            t_lognorm_mu = -0.8
            t_lognorm_sigma = 0.8
            normal = torch.randn(B, device=device, dtype=torch.float32)
            normal = normal * t_lognorm_sigma + t_lognorm_mu
            t = torch.sigmoid(normal).to(dtype=torch.bfloat16)
            
            t_reshaped = t.view(B, 1, 1, 1)
            
            # MiniT2I adds noise multiplied by noise_scale (default 2.0 in minit2i)
            noise_scale = 2.0
            noise = torch.randn_like(inputs) * noise_scale
            
            # x_t = images * t + noise * (1 - t)
            xt = inputs * t_reshaped + noise * (1.0 - t_reshaped)

            # Model outputs direct v-prediction 
            # Note: MiniT2I Wrapper's pred_velocity outputs (pred_x0 - x_t) / (1 - t)
            v_pred = model(xt, t, y_indices, y_offsets)

            # Calculate target
            target = (inputs - xt) / (1.0 - t_reshaped).clamp_min(0.05)
            
            # Compute flow matching loss per sample
            per_sample_loss = F.mse_loss(v_pred, target, reduction='none').mean(dim=(1, 2, 3))
            loss = per_sample_loss.mean()

            loss = loss / accum_steps
            loss.backward()
            loss_accum += loss.item() * accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        running_loss += loss_accum
        
        if rank == 0:
            pbar.update(1)

            if global_step % config['training']['log_every_steps'] == 0:
                log_interval = config['training']['log_every_steps']
                avg_loss = running_loss / log_interval
                wandb.log({
                    "train/loss": avg_loss,
                }, step=global_step)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                running_loss = 0.0

            if global_step % config['training']['save_image_every_steps'] == 0:
                save_checkpoint(model, optimizer, rank, config['training']['output_dir'],
                                global_step, config, fixed_prompts, fixed_noise)

                print(f"\n[Step {global_step}] Generating validation samples...")
                model.eval()
                with torch.no_grad():
                    # Generate validation samples using wrapper's euler sampler
                    y_indices_fixed, y_offsets_fixed = tag_processor.process_prompts(fixed_prompts, device)
                    
                    samples = model_raw.sample(
                        y_indices_fixed, 
                        y_offsets_fixed, 
                        image_size=config['training']['image_size'],
                        cfg_scale=config['training'].get('cfg_scale', 6.0),
                        generator=torch.Generator(device=device).manual_seed(42),
                        num_inference_steps=min(config['training'].get('n_T', 50), 50)
                    )

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
