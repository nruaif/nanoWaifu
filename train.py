import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import argparse
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image
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


def sample_ltg_timesteps(batch_size, num_patches, loc=0.0, scale=1.0, std=0.2, device="cpu", dtype=torch.float32):
    """
    Logit-Normal Truncated Gaussian (LTG) Timestep Sampler from Algorithm S2
    (Patch Forcing: Schusterbauer et al., CompVis @ LMU Munich).
    Samples per-patch timesteps where t_max ~ LogitNorm(loc, scale),
    and t_i ~ truncate(N(t_max, std_eff^2)) with std_eff = min(t_max / 2, std).
    """
    eps_max = torch.randn(batch_size, device=device)
    t_max = torch.sigmoid(loc + scale * eps_max)  # Logit-Normal
    std_eff = torch.min(t_max / 2.0, torch.full_like(t_max, std))

    t_max_2d = t_max.unsqueeze(1)
    std_eff_2d = std_eff.unsqueeze(1)

    eps = torch.randn(batch_size, num_patches, device=device)
    t = t_max_2d - torch.abs(eps) * std_eff_2d

    # Reset values < 0 to uniform in [0, t_max]
    neg_mask = (t < 0.0)
    if neg_mask.any():
        t = torch.where(neg_mask, torch.rand_like(t) * t_max_2d, t)

    return t.to(dtype)


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

    if use_cached_latents:
        data_stats_path = config['data'].get('stats_path', "runtime_stats_online.pt")
        alt_path = "runtime_stats_online.pt"
        if not os.path.exists(data_stats_path) and os.path.exists(alt_path):
            data_stats_path = alt_path
        if os.path.exists(data_stats_path):
            print(f">>> Loading cached latent data stats from {data_stats_path}...")
            dstats = torch.load(data_stats_path, map_location='cpu')
            dm = dstats["mean"]
            ds = dstats["std"]
            if dm.ndim == 1:
                dm = dm.view(1, -1, 1, 1)
            if ds.ndim == 1:
                ds = ds.view(1, -1, 1, 1)
            data_mean = dm.to(device=device, dtype=torch.bfloat16)
            data_std = ds.to(device=device, dtype=torch.bfloat16)
            print(f"    data stats: mean {dm.float().mean().item():.5f} std {ds.float().mean().item():.5f}")
        else:
            print(f">>> Warning: {data_stats_path} not found, falling back to VAE stats for cached latents")
            data_mean, data_std = latents_mean, latents_std
    else:
        data_mean, data_std = None, None

    latent_size = (image_size // 8) if (use_vae or use_tiny_vae) else image_size // config['model'].get('patch_size', 16)

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

            if "final_proj.weight" in state_dict and "final_proj.weight" in model_state:
                if state_dict["final_proj.weight"].shape != model_state["final_proj.weight"].shape:
                    old_w = state_dict["final_proj.weight"]
                    old_b = state_dict.get("final_proj.bias", None)
                    new_w = model_state["final_proj.weight"].clone()
                    new_b = model_state["final_proj.bias"].clone()
                    new_w[:old_w.shape[0]] = old_w
                    if old_b is not None:
                        new_b[:old_b.shape[0]] = old_b
                    state_dict["final_proj.weight"] = new_w
                    state_dict["final_proj.bias"] = new_b
                    print(f">>> Adapted final_proj from {old_w.shape} to {new_w.shape} for difficulty head.")

            keys_to_delete = [
                k for k in state_dict
                if k in model_state and state_dict[k].shape != model_state[k].shape
            ]
            for k in keys_to_delete:
                print(f">>> Shape Mismatch: Removing {k}. "
                      f"Checkpoint: {state_dict[k].shape}, Model: {model_state[k].shape}")
                del state_dict[k]

            model_to_load.load_state_dict(state_dict, strict=False)
            global_step = checkpoint["global_step"]

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if config['training'].get('compile', False):
        print(">>> Compiling Model...")
        model = torch.compile(model, mode="max-autotune")

    # Optimizer configuration
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
        opt_adamw = DionAdamW(adamw_params, lr=config['training']['learning_rate'], weight_decay=0, betas=(0.9, 0.999),
                              cautious_wd=True)
        opt_normuon = NorMuon(normuon_params, lr=config['training']['learning_rate'] * 10, weight_decay=0,
                              cautious_wd=True, normuon_variant=True)
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

        def _adapt_state(self, opt):
            for group in opt.param_groups:
                for p in group['params']:
                    if p in opt.state:
                        state = opt.state[p]
                        for k, v in list(state.items()):
                            if k != 'step' and torch.is_tensor(v) and v.shape != p.shape:
                                new_v = torch.zeros(p.shape, dtype=v.dtype, device=v.device)
                                slices = tuple(slice(0, min(s_old, s_new)) for s_old, s_new in zip(v.shape, p.shape))
                                new_v[slices] = v[slices]
                                state[k] = new_v
                                print(f">>> Adapted optimizer state tensor '{k}' from {v.shape} to {p.shape}")

        def load_state_dict(self, state):
            if "opt1" in state:
                self.opt1.load_state_dict(state["opt1"])
            elif "state" in state and "param_groups" in state:
                try:
                    self.opt1.load_state_dict(state)
                except Exception:
                    pass
            if "opt2" in state:
                self.opt2.load_state_dict(state["opt2"])
            self._adapt_state(self.opt1)
            self._adapt_state(self.opt2)

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
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-C2I3'), config=config)
        pbar = tqdm(range(global_step, config['training'].get('max_train_steps', 1000000)),
                    desc="Training", dynamic_ncols=True)
        os.makedirs(config['training']['output_dir'], exist_ok=True)

    if (use_vae or use_tiny_vae) and not use_cached_latents:
        vae = torch.compile(vae)

    data_iter = iter(dataloader)
    running_fm_loss = 0.0
    running_nll_loss = 0.0
    running_total_loss = 0.0
    running_infonce_loss = 0.0
    accum_steps = config['training'].get('grad_accum_steps', 1)

    while global_step < config['training'].get('max_train_steps', 1000000):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        fm_loss_accum = 0.0
        nll_loss_accum = 0.0
        total_loss_accum = 0.0
        infonce_loss_accum = 0.0

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
                if data_mean is not None and data_std is not None:
                    # Dynamically detect unnormalized shards (std ~ 0.46) vs already normalized shards (std ~ 1.0)
                    sample_std = latents.std(dim=(1, 2, 3), keepdim=True)
                    needs_norm = sample_std < 0.75
                    latents = torch.where(needs_norm, (latents - data_mean) / data_std, latents)
                inputs = latents
            elif use_tiny_vae:
                images = images.to(device, memory_format=torch.channels_last)
                with torch.no_grad():
                    v_images = images.to(dtype=torch.bfloat16)
                    out = vae.encode(v_images, return_dict=False)
                    latents = out[0] if isinstance(out, tuple) else out
                    latents = (latents - latents_mean) / latents_std
                    inputs = latents.to(dtype=torch.bfloat16)
            elif use_vae:
                images = images.to(device, memory_format=torch.channels_last)
                with torch.no_grad():
                    v_images = images.to(dtype=torch.bfloat16)
                    latents = vae.encode(v_images).latent_dist.mode()
                    latents = F.pixel_unshuffle(latents, 2).to(dtype=torch.bfloat16)
                    latents = (latents - latents_mean) / latents_std
                    inputs = latents
            else:
                images = images.to(device, memory_format=torch.channels_last)
                inputs = images.to(dtype=torch.bfloat16)

            if rank == 0 and fixed_prompts is None:
                fixed_prompts = prompts[:16]
                fixed_noise = torch.randn_like(inputs[:16])

            B, C, H, W = inputs.shape

            N = H * W

            # --- Patch Forcing (PF): Logit-Normal Truncated Gaussian (LTG) Timestep Sampler ---
            sampler_mode = config['training'].get('timestep_sampler', 'ltg')
            if sampler_mode == 'ltg':
                loc = float(config['training'].get('ltg_loc', 0.0))
                scale = float(config['training'].get('ltg_scale', 1.0))
                std = float(config['training'].get('ltg_std', 0.2))
                t = sample_ltg_timesteps(B, N, loc=loc, scale=scale, std=std, device=device, dtype=inputs.dtype)
            elif sampler_mode == 'uniform':
                t = torch.rand(B, N, device=device, dtype=inputs.dtype)
            else:
                t_scalar = torch.rand(B, device=device, dtype=inputs.dtype)
                t = t_scalar.unsqueeze(1).expand(B, N)

            t_4d = t.view(B, 1, H, W)
            noise = torch.randn_like(inputs)
            xt = (1.0 - t_4d) * inputs + t_4d * noise
            v_target = noise - inputs

            v_pred, logvar_theta, infonce_loss = model(
                xt, t, y_indices, y_offsets,
                return_logvar=True, return_layer_match=True
            )

            # Flow Matching velocity loss (Eq 2)
            fm_loss = F.mse_loss(v_pred.float(), v_target.float())

            # Difficulty-aware Heteroscedastic NLL Loss (Eq 4 in paper)
            lambda_nll = float(config['model'].get('lambda_nll', 0.01))
            if lambda_nll > 0:
                # Stop-gradient on v_pred as required by Eq 4: sg(v_theta)
                diff_sq = (v_target.float() - v_pred.detach().float()).pow(2)
                mse_per_patch = diff_sq.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                logvar_f = logvar_theta.float()
                nll_loss = 0.5 * (torch.exp(-logvar_f) * mse_per_patch + logvar_f).mean()
            else:
                nll_loss = torch.tensor(0.0, device=device)

            total_loss = fm_loss + lambda_nll * nll_loss
            if infonce_loss is not None:
                total_loss = total_loss + infonce_loss

            loss = total_loss / accum_steps
            loss.backward()

            fm_loss_accum += fm_loss.detach().item() / accum_steps
            nll_loss_accum += nll_loss.detach().item() / accum_steps
            total_loss_accum += total_loss.detach().item() / accum_steps
            if infonce_loss is not None:
                infonce_loss_accum += infonce_loss.detach().item() / accum_steps

        optimizer.step()

        global_step += 1
        running_fm_loss += fm_loss_accum
        running_nll_loss += nll_loss_accum
        running_total_loss += total_loss_accum
        running_infonce_loss += infonce_loss_accum

        if rank == 0:
            pbar.update(1)

            if global_step % config['training']['log_every_steps'] == 0:
                log_interval = config['training']['log_every_steps']
                avg_fm_loss = running_fm_loss / log_interval
                avg_nll_loss = running_nll_loss / log_interval
                avg_total_loss = running_total_loss / log_interval
                avg_infonce_loss = running_infonce_loss / log_interval

                wandb.log({
                    "train/fm_loss": avg_fm_loss,
                    "train/nll_loss": avg_nll_loss,
                    "train/total_loss": avg_total_loss,
                    "train/infonce_loss": avg_infonce_loss,
                    "train/logvar_mean": logvar_theta.detach().float().mean().item(),
                }, step=global_step)

                pbar.set_postfix({
                    "fm": f"{avg_fm_loss:.4f}",
                    "nll": f"{avg_nll_loss:.4f}",
                    "total": f"{avg_total_loss:.4f}",
                    "infonce": f"{avg_infonce_loss:.4f}",
                })

                running_fm_loss = 0.0
                running_nll_loss = 0.0
                running_total_loss = 0.0
                running_infonce_loss = 0.0

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
                        sampler_type=config['training'].get('sampler_type', 'look-ahead'),
                        p_percentile=float(config['training'].get('p_percentile', 0.4)),
                        alpha=float(config['training'].get('alpha', 2.0)),
                        inner_steps=int(config['training'].get('inner_steps', 4))
                    )

                    if use_tiny_vae:
                        samples = samples.to(dtype=torch.bfloat16)
                        if 'latents_mean' in locals() and 'latents_std' in locals() and latents_mean is not None and latents_std is not None:
                            mean = latents_mean.to(device=samples.device, dtype=samples.dtype)
                            std = latents_std.to(device=samples.device, dtype=samples.dtype)
                            latents = samples * std + mean
                        else:
                            latents = samples
                        out = vae.decode(latents, return_dict=False)
                        recon = out[0] if isinstance(out, tuple) else out
                        samples = recon.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)

                    elif use_vae:
                        samples = samples.to(dtype=torch.bfloat16)
                        if 'latents_mean' in locals() and 'latents_std' in locals() and latents_mean is not None and latents_std is not None:
                            mean = latents_mean.to(device=samples.device, dtype=samples.dtype)
                            std = latents_std.to(device=samples.device, dtype=samples.dtype)
                            latents = samples * std + mean
                        else:
                            latents = samples * 0.82
                        latents = F.pixel_shuffle(latents, 2)
                        recon = vae.decode(latents).sample
                        samples = recon.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)

                    grid = make_grid(samples, nrow=4)
                    grid = grid.clamp(0, 1)
                    grid_pil = TF.to_pil_image(grid.cpu())
                    wandb.log({
                        "val/samples": wandb.Image(grid_pil, caption=f"Step {global_step}")
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