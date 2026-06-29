import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import os
import argparse
import random
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
import glob
import builtins
import numpy as np
from dataset import WDSLoader
import torch.nn.functional as F
from huggingface_hub import upload_file
import threading

from adversarial_flow import (
    AdversarialFlowDiscriminator,
    GradientNormalization,
    cosine_decay,
    discriminator_losses,
    generator_losses,
    interpolate_flow,
    sample_adversarial_flow,
    sample_afm_timesteps,
    set_requires_grad,
)


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


def _linear_sum_assignment(cost):
    """Exact O(n^3) Hungarian assignment for a square NumPy cost matrix."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("OT cost matrix must be square")

    size = cost.shape[0]
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    matching = np.zeros(size + 1, dtype=np.int64)
    path = np.zeros(size + 1, dtype=np.int64)

    for row in range(1, size + 1):
        matching[0] = row
        min_cost = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column = 0

        while True:
            used[column] = True
            current_row = matching[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[current_row - 1, candidate - 1]
                    - u[current_row]
                    - v[candidate]
                )
                if reduced < min_cost[candidate]:
                    min_cost[candidate] = reduced
                    path[candidate] = column
                if min_cost[candidate] < delta:
                    delta = min_cost[candidate]
                    next_column = candidate

            for candidate in range(size + 1):
                if used[candidate]:
                    u[matching[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_cost[candidate] -= delta

            column = next_column
            if matching[column] == 0:
                break

        while True:
            previous = path[column]
            matching[column] = matching[previous]
            column = previous
            if column == 0:
                break

    assignment = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        assignment[matching[column] - 1] = column - 1
    return assignment


def minibatch_ot_pair_noise(inputs, noise):
    """
    Hard minibatch OT pairing using squared Euclidean cost.

    inputs: [B, C, H, W] data/latent batch
    noise:  [B, C, H, W] Gaussian batch

    Returns:
        noise reordered to match inputs by OT assignment.
    """
    B = inputs.shape[0]

    with torch.no_grad():
        x = inputs.detach().float().flatten(1)
        z = noise.detach().float().flatten(1)
        cost = (
            x.square().sum(dim=1, keepdim=True)
            + z.square().sum(dim=1).unsqueeze(0)
            - 2.0 * x @ z.T
        ).clamp_min_(0.0)
        assignment = _linear_sum_assignment(cost.cpu().numpy())
        col_ind = torch.as_tensor(
            assignment,
            device=inputs.device,
            dtype=torch.long,
        )

    return noise[col_ind]


def unwrap_model(model):
    while True:
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def resolve_checkpoint_path(path):
    if not path:
        return None
    if os.path.isdir(path):
        checkpoint_files = glob.glob(
            os.path.join(path, "ckpt_step_*.pth")
        )
        if not checkpoint_files:
            return None
        return max(
            checkpoint_files,
            key=lambda item: int(
                os.path.basename(item).split("_")[-1].split(".")[0]
            ),
        )
    return path if os.path.exists(path) else None


def load_checkpoint(path, map_location):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)
    except RuntimeError as error:
        if "mmap" not in str(error).lower():
            raise
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )


def sample_training_timesteps(batch_size, device, minimum=1e-3):
    if not 0.0 <= minimum < 0.5:
        raise ValueError("training.timestep_min must be in [0, 0.5)")
    timesteps = torch.rand(batch_size, device=device, dtype=torch.float32)
    return timesteps * (1.0 - 2.0 * minimum) + minimum


def x0_loss_weight(timesteps, snr_gamma=5.0):
    timesteps = timesteps.float().view(-1, 1, 1, 1)
    snr = (1.0 - timesteps).square() / timesteps.square().clamp_min(1e-6)
    if snr_gamma is not None and snr_gamma > 0:
        snr = snr.clamp(max=float(snr_gamma))
    return snr


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
    if max_checkpoints is None or max_checkpoints <= 0:
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
        push_to_hf=False,
        repo_id=None,
        discriminator=None,
        discriminator_optimizer=None,
        gradient_normalizer=None,
        training_mode="flow_matching",
):
    if rank != 0:
        return

    print(f"\n[Step {step}] Saving Checkpoint...")

    os.makedirs(output_dir, exist_ok=True)
    model_to_save = unwrap_model(model)
    ckpt_path = os.path.join(output_dir, f'ckpt_step_{step}.pth')

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        # FIX: save optimizer state so resuming doesn't lose momentum/adaptive stats
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step,
        "config": config,
        "training_mode": training_mode,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "fixed_prompts": fixed_prompts,
        "fixed_noise": fixed_noise.cpu() if fixed_noise is not None else None,
    }
    if discriminator is not None:
        checkpoint["discriminator_state_dict"] = unwrap_model(
            discriminator
        ).state_dict()
    if discriminator_optimizer is not None:
        checkpoint["discriminator_optimizer_state_dict"] = (
            discriminator_optimizer.state_dict()
        )
    if gradient_normalizer is not None:
        checkpoint["gradient_normalizer_state_dict"] = (
            gradient_normalizer.state_dict()
        )

    torch.save(checkpoint, ckpt_path)
    training_config = config.get('training', config)
    cleanup_checkpoints(output_dir, training_config.get('max_checkpoints', 3), rank)
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
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    adversarial_config = config.get('adversarial', {})
    adversarial_enabled = bool(adversarial_config.get('enabled', False))

    seed = int(training_config.get('seed', 1337))
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)

    from model_dit import TokenformerDiT, TagProcessor, sample_flow
    ModelClass, TagProcessor, sample_fn = TokenformerDiT, TagProcessor, sample_flow

    tag_processor = TagProcessor("tags.txt")
    num_classes = tag_processor.num_classes

    use_cached_latents = data_config.get('use_cached_latents', False)
    cached_latent_scale = data_config.get('cached_latent_scale', 1.0)
    if cached_latent_scale is None:
        cached_latent_scale = 1.0
    cached_latent_scale = float(cached_latent_scale)
    if cached_latent_scale <= 0:
        raise ValueError("data.cached_latent_scale must be positive")
    if use_cached_latents:
        print(f">>> Using Cached Latents mode (loading pre-computed latents from {data_config['cache_dir']})...")
        url = data_config['cache_dir']
        wds_loader = WDSLoader(
            url=url,
            csv_path=data_config.get('csv_path'),
            image_size=training_config['image_size'],
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            use_advanced_captions=data_config.get('use_advanced_captions', True)
        )
        dataloader = wds_loader.make_loader()
    else:
        wds_loader = WDSLoader(
            url=data_config['webdataset_url'],
            csv_path=data_config.get('csv_path'),
            image_size=training_config['image_size'],
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            use_advanced_captions=data_config.get('use_advanced_captions', True)
        )
        dataloader = wds_loader.make_loader()

    image_size = training_config['image_size']

    use_vae = model_config.get('use_vae', False)
    use_tiny_vae = model_config.get('use_tiny_vae', False)
    in_channels = model_config.get('in_channels', 3)

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

    latent_downsample = 16 if (use_vae or use_tiny_vae) else model_config.get('patch_size', 16)
    latent_size = image_size // latent_downsample

    model = ModelClass(
        in_channels=in_channels,
        dim=model_config.get('dim', 768),
        depth=model_config.get('depth', 12),
        num_heads=model_config.get('num_heads', 12),
        num_classes=num_classes,
        use_checkpoint=training_config.get('gradient_checkpointing', False),
    ).to(device=device, dtype=torch.bfloat16)

    discriminator = None
    gradient_normalizer = None
    if adversarial_enabled:
        afm_steps = adversarial_config.get('steps', 1)
        if not isinstance(afm_steps, int) or afm_steps < 1:
            raise ValueError(
                "This DiT supports designated AFM steps only; "
                "adversarial.steps must be a positive integer"
            )
        discriminator = AdversarialFlowDiscriminator(
            in_channels=in_channels,
            dim=int(adversarial_config.get('discriminator_dim', 512)),
            depth=int(adversarial_config.get('discriminator_depth', 8)),
            num_heads=int(
                adversarial_config.get('discriminator_heads', 8)
            ),
            num_classes=num_classes,
            use_checkpoint=bool(
                adversarial_config.get(
                    'discriminator_gradient_checkpointing',
                    False,
                )
            ),
        ).to(device=device)
        gradient_normalizer = GradientNormalization(
            ema_decay=float(
                adversarial_config.get('gradient_norm_decay', 0.9)
            ),
        ).to(device)

    # A resume continues the same training mode. init_from only imports weights.
    global_step = 0
    resume_path = resolve_checkpoint_path(
        training_config.get('resume_from', config.get('resume_from'))
    )
    init_path = resolve_checkpoint_path(training_config.get('init_from'))
    resume_checkpoint = None
    fixed_prompts = None
    fixed_noise = None

    checkpoint_path = resume_path or init_path
    if checkpoint_path:
        is_resume = resume_path is not None
        action = "Resuming from" if is_resume else "Initializing model from"
        print(f"{action}: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        expected_mode = (
            "adversarial_flow" if adversarial_enabled else "flow_matching"
        )
        checkpoint_mode = checkpoint.get(
            "training_mode",
            "flow_matching",
        )
        if is_resume and checkpoint_mode != expected_mode:
            raise ValueError(
                f"Cannot resume {expected_mode} from a {checkpoint_mode} "
                "checkpoint. Use training.init_from to import model weights."
            )

        model_to_load = unwrap_model(model)
        state_dict = dict(checkpoint["model_state_dict"])
        model_state = model_to_load.state_dict()
        keys_to_delete = [
            key
            for key in state_dict
            if key in model_state
            and state_dict[key].shape != model_state[key].shape
        ]
        for key in keys_to_delete:
            print(
                f">>> Shape Mismatch: Removing {key}. "
                f"Checkpoint: {state_dict[key].shape}, "
                f"Model: {model_state[key].shape}"
            )
            del state_dict[key]

        incompatible = model_to_load.load_state_dict(
            state_dict,
            strict=False,
        )
        if incompatible.missing_keys:
            print(
                f">>> Missing model keys: {len(incompatible.missing_keys)}"
            )
        if incompatible.unexpected_keys:
            print(
                ">>> Unexpected checkpoint keys: "
                f"{len(incompatible.unexpected_keys)}"
            )

        if is_resume:
            resume_checkpoint = checkpoint
            global_step = max(0, int(checkpoint.get("global_step", 0)))
            if checkpoint.get("fixed_noise") is not None:
                fixed_noise = checkpoint["fixed_noise"]
            if "fixed_prompts" in checkpoint:
                fixed_prompts = checkpoint["fixed_prompts"]
            if checkpoint.get("rng_state") is not None:
                torch.set_rng_state(checkpoint["rng_state"].cpu())
            if checkpoint.get("python_rng_state") is not None:
                random.setstate(checkpoint["python_rng_state"])
            if checkpoint.get("numpy_rng_state") is not None:
                np.random.set_state(checkpoint["numpy_rng_state"])
            if (
                torch.cuda.is_available()
                and checkpoint.get("cuda_rng_state") is not None
            ):
                torch.cuda.set_rng_state(
                    checkpoint["cuda_rng_state"].cpu(),
                    device=device,
                )

            if adversarial_enabled:
                if "discriminator_state_dict" not in checkpoint:
                    raise ValueError(
                        "AFM resume checkpoint has no discriminator state"
                    )
                discriminator.load_state_dict(
                    checkpoint["discriminator_state_dict"],
                    strict=True,
                )
                if checkpoint.get(
                    "gradient_normalizer_state_dict"
                ) is not None:
                    gradient_normalizer.load_state_dict(
                        checkpoint[
                            "gradient_normalizer_state_dict"
                        ],
                        strict=True,
                    )
        del state_dict
        del model_state
        if not is_resume:
            del checkpoint

    if training_config.get('compile', False):
        print(">>> Compiling Model...")
        model = torch.compile(model, mode="max-autotune")
    if adversarial_enabled and adversarial_config.get('compile', False):
        print(">>> Compiling Discriminator...")
        discriminator = torch.compile(
            discriminator,
            mode="max-autotune",
        )

    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )
        if discriminator is not None:
            discriminator = DDP(
                discriminator,
                device_ids=[local_rank],
                find_unused_parameters=False,
            )
    use_afm_adamw = (
        adversarial_enabled
        and adversarial_config.get(
            'generator_optimizer',
            'adamw',
        ).lower()
        == 'adamw'
    )
    if use_afm_adamw:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(
                adversarial_config.get(
                    'generator_learning_rate',
                    training_config['learning_rate'],
                )
            ),
            betas=tuple(adversarial_config.get('betas', (0.0, 0.9))),
            eps=float(adversarial_config.get('eps', 1e-8)),
            weight_decay=float(
                adversarial_config.get('weight_decay', 0.01)
            ),
        )
    else:
        try:
            from adv_optm import Muon_adv as NorMuon
            from adv_optm import AdamW_adv as DionAdamW
        except ImportError:
            print(
                "Warning: Advanced optimizers missing, "
                "falling back to torch.optim.AdamW"
            )
            NorMuon, DionAdamW = torch.optim.AdamW, torch.optim.AdamW

        adamw_params, normuon_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_1d = parameter.ndim <= 1
            is_dwconv = (
                'dwconv' in name
                or 'dw_conv' in name
                or (
                    parameter.ndim == 4
                    and parameter.shape[1] == 1
                )
            )
            is_embedding = (
                'embed' in name or 'token_embed' in name
            )
            if is_1d or is_dwconv or is_embedding:
                adamw_params.append(parameter)
            else:
                normuon_params.append(parameter)

        try:
            opt_adamw = DionAdamW(
                adamw_params,
                lr=training_config['learning_rate'],
                weight_decay=0,
                betas=(0.9, 0.95),
                cautious_wd=True,
            )
            opt_normuon = NorMuon(
                normuon_params,
                lr=training_config['learning_rate'] * 10,
                weight_decay=0.1,
                cautious_wd=True,
                normuon_variant=True,
            )
        except Exception:
            opt_adamw = torch.optim.AdamW(
                adamw_params,
                lr=training_config['learning_rate'],
                weight_decay=0.1,
            )
            opt_normuon = torch.optim.AdamW(
                normuon_params,
                lr=training_config['learning_rate'],
                weight_decay=0.1,
            )

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
                return {
                    "opt1": self.opt1.state_dict(),
                    "opt2": self.opt2.state_dict(),
                }

            def load_state_dict(self, state):
                if "opt1" in state:
                    self.opt1.load_state_dict(state["opt1"])
                if "opt2" in state:
                    self.opt2.load_state_dict(state["opt2"])

        optimizer = DualOptimizer(opt_adamw, opt_normuon)
    discriminator_optimizer = None
    if adversarial_enabled:
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(
                adversarial_config.get(
                    'discriminator_learning_rate',
                    training_config['learning_rate'],
                )
            ),
            betas=tuple(adversarial_config.get('betas', (0.0, 0.9))),
            eps=float(adversarial_config.get('eps', 1e-8)),
            weight_decay=float(
                adversarial_config.get('weight_decay', 0.01)
            ),
        )

    if resume_checkpoint is not None and "optimizer_state_dict" in resume_checkpoint:
        try:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            print(">>> Optimizer state restored.")
        except Exception as e:
            print(f">>> Could not restore optimizer state: {e}. Starting fresh.")
    if (
        adversarial_enabled
        and resume_checkpoint is not None
        and "discriminator_optimizer_state_dict" in resume_checkpoint
    ):
        discriminator_optimizer.load_state_dict(
            resume_checkpoint["discriminator_optimizer_state_dict"]
        )
        print(">>> Discriminator optimizer state restored.")
    if resume_checkpoint is not None:
        resume_checkpoint = None
        del checkpoint

    if rank == 0:
        wandb.init(project=training_config.get('wandb_project', config.get('wandb_project', 'nanoWaifu-C2I')), config=config)
        pbar = tqdm(range(global_step, training_config.get('max_train_steps', 1000000)),
                    desc="Training", dynamic_ncols=True)
        os.makedirs(training_config['output_dir'], exist_ok=True)
    if (use_vae or use_tiny_vae) and not use_cached_latents:
        vae = torch.compile(vae)
    data_iter = iter(dataloader)

    def next_batch(iterator):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        return iterator, batch

    def prepare_batch(batch):
        images, prompts, _ = batch
        y_indices, y_offsets = tag_processor.process_prompts(
            prompts,
            device,
            dropout_prob=training_config.get(
                'class_dropout_prob',
                0.1,
            ),
        )

        if use_cached_latents:
            latents = images.to(device=device, dtype=torch.bfloat16)
            inputs = latents / cached_latent_scale
        elif use_tiny_vae:
            images = images.to(
                device,
                memory_format=torch.channels_last,
            )
            with torch.no_grad():
                encoded = vae.encode(
                    images.to(dtype=torch.bfloat16),
                    return_dict=False,
                )
                latents = (
                    encoded[0] if isinstance(encoded, tuple) else encoded
                )
                inputs = (
                    (latents - latents_mean) / latents_std
                ).to(dtype=torch.bfloat16)
        elif use_vae:
            images = images.to(
                device,
                memory_format=torch.channels_last,
            )
            with torch.no_grad():
                latents = vae.encode(
                    images.to(dtype=torch.bfloat16)
                ).latent_dist.mode()
                latents = F.pixel_unshuffle(
                    latents,
                    2,
                ).to(dtype=torch.bfloat16)
                inputs = (latents - latents_mean) / latents_std
        else:
            images = images.to(
                device,
                memory_format=torch.channels_last,
            )
            inputs = images.to(dtype=torch.bfloat16)

        return inputs, prompts, y_indices, y_offsets

    def decode_samples(samples):
        if use_tiny_vae:
            latents = samples.to(dtype=torch.bfloat16)
            if use_cached_latents:
                latents = latents * cached_latent_scale
            latents = latents * latents_std + latents_mean
            decoded = vae.decode(latents, return_dict=False)
            samples = (
                decoded[0] if isinstance(decoded, tuple) else decoded
            )
            return samples.clamp(-1, 1).float() / 2.0 + 0.5

        if use_vae:
            latents = samples.to(dtype=torch.bfloat16)
            if use_cached_latents:
                latents = latents * cached_latent_scale
            latents = latents * latents_std + latents_mean
            latents = F.pixel_shuffle(latents, 2)
            samples = vae.decode(latents).sample
            return samples.clamp(-1, 1).float() / 2.0 + 0.5

        return samples.float().clamp(-1, 1) / 2.0 + 0.5

    def save_current_checkpoint(training_mode):
        save_checkpoint(
            model,
            optimizer,
            rank,
            training_config['output_dir'],
            global_step,
            config,
            fixed_prompts,
            fixed_noise,
            push_to_hf=training_config.get('push_to_hf', False),
            repo_id=training_config.get('hf_repo_id'),
            discriminator=discriminator if adversarial_enabled else None,
            discriminator_optimizer=(
                discriminator_optimizer if adversarial_enabled else None
            ),
            gradient_normalizer=(
                gradient_normalizer if adversarial_enabled else None
            ),
            training_mode=training_mode,
        )

    if adversarial_enabled:
        afm_steps = int(adversarial_config.get('steps', 1))
        discriminator_steps = int(
            adversarial_config.get('discriminator_steps', 1)
        )
        if discriminator_steps < 1:
            raise ValueError(
                "adversarial.discriminator_steps must be positive"
            )

        gp_ratio = float(
            adversarial_config.get('gp_batch_ratio', 0.25)
        )
        if not 0.0 < gp_ratio <= 1.0:
            raise ValueError(
                "adversarial.gp_batch_ratio must be in (0, 1]"
            )
        gp_eps = float(adversarial_config.get('gp_epsilon', 0.01))
        if gp_eps <= 0:
            raise ValueError(
                "adversarial.gp_epsilon must be positive"
            )

        accum_steps = int(
            training_config.get('grad_accum_steps', 1)
        )
        max_steps = int(
            training_config.get('max_train_steps', 1000000)
        )
        log_interval = int(training_config.get('log_every_steps', 10))
        running = {
            "dis_total": 0.0,
            "dis_adv": 0.0,
            "dis_r1": 0.0,
            "dis_r2": 0.0,
            "dis_center": 0.0,
            "gen_total": 0.0,
            "gen_adv": 0.0,
            "gen_ot": 0.0,
            "gen_fm": 0.0,
        }

        while global_step < max_steps:
            model.train()
            discriminator.train()
            dis_metrics = {key: 0.0 for key in running if key.startswith("dis_")}

            set_requires_grad(model, False)
            set_requires_grad(discriminator, True)
            for _ in range(discriminator_steps):
                discriminator_optimizer.zero_grad(set_to_none=True)
                for _ in range(accum_steps):
                    data_iter, batch = next_batch(data_iter)
                    (
                        inputs,
                        prompts,
                        y_indices,
                        y_offsets,
                    ) = prepare_batch(batch)
                    batch_size = len(inputs)
                    timesteps_src, timesteps_tgt = sample_afm_timesteps(
                        batch_size,
                        afm_steps,
                        device,
                    )
                    source = interpolate_flow(
                        inputs,
                        torch.randn_like(inputs),
                        timesteps_src,
                    )
                    target = interpolate_flow(
                        inputs,
                        torch.randn_like(inputs),
                        timesteps_tgt,
                    )
                    with torch.no_grad():
                        predicted = model(
                            source,
                            timesteps_src,
                            y_indices,
                            y_offsets,
                        )

                    target_for_discriminator = target.float()
                    predicted_for_discriminator = predicted.float()
                    gp_batch = max(round(batch_size * gp_ratio), 1)
                    target_gp = (
                        target_for_discriminator[:gp_batch]
                        + gp_eps
                        * torch.randn_like(
                            target_for_discriminator[:gp_batch]
                        )
                    )
                    predicted_gp = (
                        predicted_for_discriminator[:gp_batch]
                        + gp_eps
                        * torch.randn_like(
                            predicted_for_discriminator[:gp_batch]
                        )
                    )
                    logits = discriminator(
                        torch.cat(
                            [
                                target_for_discriminator,
                                predicted_for_discriminator,
                                target_gp,
                                predicted_gp,
                            ]
                        ),
                        y_indices,
                        y_offsets,
                        timesteps_tgt,
                        condition_repeats=(
                            batch_size,
                            batch_size,
                            gp_batch,
                            gp_batch,
                        ),
                    )
                    (
                        logits_real,
                        logits_fake,
                        logits_real_gp,
                        logits_fake_gp,
                    ) = logits.split(
                        [
                            batch_size,
                            batch_size,
                            gp_batch,
                            gp_batch,
                        ]
                    )
                    weighting = (
                        timesteps_src - timesteps_tgt
                    ).abs().clamp_min(0.001)
                    losses = discriminator_losses(
                        logits_real,
                        logits_fake,
                        logits_real_gp,
                        logits_fake_gp,
                        weighting,
                        gp_scale=float(
                            adversarial_config.get('gp_scale', 0.25)
                        ),
                        gp_eps=gp_eps,
                        center_scale=float(
                            adversarial_config.get(
                                'center_penalty',
                                0.01,
                            )
                        ),
                    )
                    (losses["total"] / accum_steps).backward()
                    metric_scale = 1.0 / (
                        accum_steps * discriminator_steps
                    )
                    for key in dis_metrics:
                        loss_key = key.removeprefix("dis_")
                        dis_metrics[key] += (
                            losses[loss_key].detach().item()
                            * metric_scale
                        )

                dis_clip = adversarial_config.get(
                    'discriminator_grad_clip_norm',
                    50.0,
                )
                if dis_clip is not None and dis_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(),
                        float(dis_clip),
                    )
                discriminator_optimizer.step()

            set_requires_grad(model, True)
            set_requires_grad(discriminator, False)
            optimizer.zero_grad(set_to_none=True)
            gen_metrics = {key: 0.0 for key in running if key.startswith("gen_")}
            ot_scale = cosine_decay(
                global_step,
                float(adversarial_config.get('ot_scale', 0.2)),
                float(adversarial_config.get('ot_scale_end', 0.005)),
                int(adversarial_config.get('ot_decay_steps', 1000000)),
            )
            fm_weight = float(
                adversarial_config.get('flow_matching_weight', 0.0)
            )

            for _ in range(accum_steps):
                data_iter, batch = next_batch(data_iter)
                (
                    inputs,
                    prompts,
                    y_indices,
                    y_offsets,
                ) = prepare_batch(batch)
                batch_size, channels, height, width = inputs.shape

                if rank == 0:
                    if fixed_prompts is None:
                        fixed_prompts = prompts[:16]
                        fixed_noise = torch.randn_like(
                            inputs[:len(fixed_prompts)]
                        )
                    elif (
                        fixed_noise is None
                        or fixed_noise.shape[0] != len(fixed_prompts)
                        or tuple(fixed_noise.shape[1:])
                        != (channels, height, width)
                    ):
                        fixed_noise = torch.randn(
                            len(fixed_prompts),
                            channels,
                            height,
                            width,
                            device=device,
                            dtype=inputs.dtype,
                        )
                    else:
                        fixed_noise = fixed_noise.to(
                            device=device,
                            dtype=inputs.dtype,
                        )

                timesteps_src, timesteps_tgt = sample_afm_timesteps(
                    batch_size,
                    afm_steps,
                    device,
                )
                source = interpolate_flow(
                    inputs,
                    torch.randn_like(inputs),
                    timesteps_src,
                )
                target = interpolate_flow(
                    inputs,
                    torch.randn_like(inputs),
                    timesteps_tgt,
                )
                predicted = model(
                    source,
                    timesteps_src,
                    y_indices,
                    y_offsets,
                )
                predicted_for_discriminator = (
                    gradient_normalizer(predicted.float())
                    if adversarial_config.get(
                        'gradient_normalization',
                        True,
                    )
                    else predicted.float()
                )
                logits_real, logits_fake = discriminator(
                    torch.cat(
                        [target.float(), predicted_for_discriminator]
                    ),
                    y_indices,
                    y_offsets,
                    timesteps_tgt,
                    condition_repeats=(batch_size, batch_size),
                ).split(batch_size)
                weighting = (
                    timesteps_src - timesteps_tgt
                ).abs().clamp_min(0.001)
                losses = generator_losses(
                    logits_real,
                    logits_fake,
                    predicted,
                    source,
                    weighting,
                    ot_scale,
                )
                fm_loss = F.mse_loss(
                    predicted.float(),
                    target.float(),
                )
                total_loss = losses["total"] + fm_weight * fm_loss
                (total_loss / accum_steps).backward()

                gen_metrics["gen_total"] += (
                    total_loss.detach().item() / accum_steps
                )
                gen_metrics["gen_adv"] += (
                    losses["adv"].detach().item() / accum_steps
                )
                gen_metrics["gen_ot"] += (
                    losses["ot"].detach().item() / accum_steps
                )
                gen_metrics["gen_fm"] += (
                    fm_loss.detach().item() / accum_steps
                )

            grad_clip = training_config.get('grad_clip_norm', 1.0)
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(grad_clip),
                )
            optimizer.step()
            global_step += 1

            for key, value in {**dis_metrics, **gen_metrics}.items():
                running[key] += value

            if rank == 0:
                pbar.update(1)
                if global_step % log_interval == 0:
                    metrics = {
                        f"train/{key}": value / log_interval
                        for key, value in running.items()
                    }
                    metrics["train/ot_scale"] = ot_scale
                    wandb.log(metrics, step=global_step)
                    pbar.set_postfix(
                        {
                            "d": f"{metrics['train/dis_total']:.4f}",
                            "g": f"{metrics['train/gen_total']:.4f}",
                            "ot": f"{metrics['train/gen_ot']:.4f}",
                        }
                    )
                    running = {key: 0.0 for key in running}

                if (
                    global_step
                    % training_config['save_image_every_steps']
                    == 0
                ):
                    save_current_checkpoint("adversarial_flow")
                    print(
                        f"\n[Step {global_step}] "
                        "Generating AFM validation samples..."
                    )
                    raw_model = unwrap_model(model)
                    samples = sample_adversarial_flow(
                        raw_model,
                        tag_processor,
                        latent_size,
                        len(fixed_prompts),
                        fixed_prompts,
                        device,
                        steps=afm_steps,
                        guidance_scale=training_config.get(
                            'cfg_scale',
                            1.0,
                        ),
                        noise=fixed_noise,
                    )
                    grid = make_grid(decode_samples(samples), nrow=4)
                    wandb.log(
                        {
                            "val/samples": wandb.Image(
                                grid,
                                caption=f"Step {global_step}",
                            )
                        },
                        step=global_step,
                    )

        set_requires_grad(discriminator, True)
        if rank == 0:
            save_current_checkpoint("adversarial_flow")
            wandb.finish()
        cleanup_ddp()
        return

    running_fm_loss = 0.0
    running_neg_loss = 0.0
    running_deltafm_loss = 0.0
    running_total_loss = 0.0
    running_cos_loss = 0.0
    accum_steps = training_config.get('grad_accum_steps', 1)
    while global_step < training_config.get('max_train_steps', 1000000):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        fm_loss_accum = 0.0
        neg_loss_accum = 0.0
        deltafm_loss_accum = 0.0
        total_loss_accum = 0.0
        cos_loss_accum = 0.0

        for _ in range(accum_steps):
            data_iter, batch = next_batch(data_iter)
            inputs, prompts, y_indices, y_offsets = prepare_batch(batch)

            B, C, H, W = inputs.shape

            if rank == 0:
                if fixed_prompts is None:
                    fixed_prompts = prompts[:16]
                    fixed_noise = torch.randn_like(inputs[:len(fixed_prompts)])
                elif (
                    fixed_noise is None
                    or fixed_noise.shape[0] != len(fixed_prompts)
                    or tuple(fixed_noise.shape[1:]) != (C, H, W)
                ):
                    fixed_noise = torch.randn(
                        len(fixed_prompts), C, H, W, device=device, dtype=inputs.dtype
                    )
                else:
                    fixed_noise = fixed_noise.to(device=device, dtype=inputs.dtype)

            t = sample_training_timesteps(
                B,
                device,
                minimum=float(
                    training_config.get('timestep_min', 1e-3)
                ),
            )

            t_reshaped = t.to(inputs.dtype).view(B, 1, 1, 1)
            noise = torch.randn_like(inputs)
            if training_config.get('use_minibatch_ot', False):
                noise = minibatch_ot_pair_noise(inputs, noise)
            xt = (1 - t_reshaped) * inputs + t_reshaped * noise

            # --- Training Augmentations (each applied independently w/ 50% chance) ---
            # 1. Gaussian noise injection to xt to simulate drift during inference
            noise_inject_ratio = training_config.get('noise_inject_ratio', 0.1)
            if noise_inject_ratio > 0:
                noise_mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5).to(xt.dtype)
                noise_injection = torch.randn_like(xt)
                xt = xt + noise_mask * noise_inject_ratio * noise_injection

            # 2. Intra-sample crossing: build xt_neg at the SAME timestep t
            #    from a different clean sample to simulate mean-seeking drift
            cross_ratio = training_config.get('cross_ratio', 0.1)
            if cross_ratio > 0:
                cross_mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5).to(xt.dtype)
                inputs_neg = inputs.roll(shifts=1, dims=0)
                noise_neg = torch.randn_like(inputs)

                xt_neg = (1 - t_reshaped) * inputs_neg + t_reshaped * noise_neg
                xt = xt + cross_mask * cross_ratio * (xt_neg - xt)

            x0_pred = model(xt, t, y_indices, y_offsets)

            # Compute x0-space loss with SNR weighting
            x0_target = inputs
            snr = x0_loss_weight(
                t,
                training_config.get('snr_gamma', 5.0),
            )
            
            deltafm_lambda = training_config.get('deltafm_lambda', 0.05)
            if deltafm_lambda > 0 and B > 1:
                perm = torch.arange(B, device=device).roll(1)
            
                inputs_neg = inputs[perm]
            
                fm_loss = (F.mse_loss(x0_pred.float(), x0_target.float(), reduction='none') * snr).mean()
                neg_loss = (F.mse_loss(x0_pred.float(), inputs_neg.float(), reduction='none') * snr).mean()
            
                # ∆FM flow objective
                deltafm_loss = fm_loss - deltafm_lambda * neg_loss
            else:
                fm_loss = (F.mse_loss(x0_pred.float(), x0_target.float(), reduction='none') * snr).mean()
                neg_loss = torch.zeros((), device=device)
                deltafm_loss = fm_loss
            
            cos_loss = 1.0 - F.cosine_similarity(
                x0_pred.float().flatten(1),
                x0_target.float().flatten(1),
                dim=1,
                eps=1e-6,
            ).mean()
            # Full training loss
            total_loss = deltafm_loss + cos_loss

            loss = total_loss / accum_steps
            loss.backward()
            fm_loss_accum += fm_loss.detach().item() / accum_steps
            neg_loss_accum += neg_loss.detach().item() / accum_steps
            deltafm_loss_accum += deltafm_loss.detach().item() / accum_steps
            total_loss_accum += total_loss.detach().item() / accum_steps
            cos_loss_accum += cos_loss.detach().item() / accum_steps

        grad_clip_norm = training_config.get('grad_clip_norm', 1.0)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip_norm),
            )
        optimizer.step()

        global_step += 1
        running_fm_loss += fm_loss_accum
        running_neg_loss += neg_loss_accum
        running_deltafm_loss += deltafm_loss_accum
        running_total_loss += total_loss_accum
        running_cos_loss += cos_loss_accum
        if rank == 0:
            pbar.update(1)

            if global_step % training_config['log_every_steps'] == 0:
                log_interval = training_config['log_every_steps']
            
                avg_fm_loss = running_fm_loss / log_interval
                avg_neg_loss = running_neg_loss / log_interval
                avg_deltafm_loss = running_deltafm_loss / log_interval
                avg_total_loss = running_total_loss / log_interval
                avg_cos_loss = running_cos_loss / log_interval
            
                wandb.log({
                    "train/fm_loss": avg_fm_loss,
                    "train/neg_loss": avg_neg_loss,
                    "train/deltafm_loss": avg_deltafm_loss,
                    "train/total_loss": avg_total_loss,
                    "train/cos_loss": avg_cos_loss,
                    "train/deltafm_lambda": training_config.get('deltafm_lambda', 0.05),
                }, step=global_step)
            
                pbar.set_postfix({
                    "fm": f"{avg_fm_loss:.4f}",
                    "dfm": f"{avg_deltafm_loss:.4f}",
                    "total": f"{avg_total_loss:.4f}",
                    "cos": f"{avg_cos_loss:.4f}",
                })
            
                running_fm_loss = 0.0
                running_neg_loss = 0.0
                running_deltafm_loss = 0.0
                running_total_loss = 0.0
                running_cos_loss = 0.0

            if global_step % training_config['save_image_every_steps'] == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    rank,
                    training_config['output_dir'],
                    global_step,
                    config,
                    fixed_prompts,
                    fixed_noise,
                    push_to_hf=training_config.get('push_to_hf', False),
                    repo_id=training_config.get('hf_repo_id'),
                )

                print(f"\n[Step {global_step}] Generating validation samples...")
                model.eval()
                with torch.no_grad():
                    samples = sample_fn(
                        unwrap_model(model),
                        tag_processor,
                        latent_size,
                        len(fixed_prompts),
                        fixed_prompts,
                        device,
                        guidance_scale=training_config.get('cfg_scale', 1.4),
                        noise=fixed_noise,
                        sampler_type="euler"
                    )

                    if use_tiny_vae:
                        latents = samples.to(dtype=torch.bfloat16)
                        if use_cached_latents:
                            latents = latents * cached_latent_scale
                        latents = latents * latents_std + latents_mean
                        out = vae.decode(latents, return_dict=False)
                        recon = out[0] if isinstance(out, tuple) else out
                        samples = recon.clamp(-1, 1) / 2.0 + 0.5
                        samples = samples.to(dtype=torch.float32)

                    elif use_vae:
                        latents = samples.to(dtype=torch.bfloat16)
                        if use_cached_latents:
                            latents = latents * cached_latent_scale
                        latents = latents * latents_std + latents_mean
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
        save_checkpoint(
            model,
            optimizer,
            rank,
            training_config['output_dir'],
            global_step,
            config,
            fixed_prompts,
            fixed_noise,
            push_to_hf=training_config.get('push_to_hf', False),
            repo_id=training_config.get('hf_repo_id'),
        )
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
    cleanup_ddp()
