'''
Author: Xu Pan && panxurs@whu.edu.cn
Date: 2025-11-04 11:54:40
LastEditors: px panxurs@whu.edu.cn
LastEditTime: 2025-11-10 15:36:27
FilePath: /gamma/scripts/train_gamma.py
Description: 

Copyright© (c) 2025 by Xu Pan, All Rights Reserved. 
'''
import re
import gc
import os
import jax
import time as pytime
import tqdm
import wandb
import torch
import shutil
import logging
import platform
import dataclasses
import numpy as np
import safetensors.torch
import torch.nn.parallel
import torch.distributed as dist
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.shared.normalize as _normalize


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # [MODIFIED] rank-aware filter with opt-in for all ranks via env
    class RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            allow_all = os.environ.get("OPENPI_LOG_ALL_RANKS", "0") == "1"
            if allow_all:
                return True
            try:
                rank = int(os.environ.get("RANK", "0"))
            except Exception:
                rank = 0
            return (rank == 0) or (record.levelno >= logging.WARNING)

    # [ADDED] tqdm-friendly logging handler
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record)
            except Exception:
                msg = record.getMessage()
            tqdm.tqdm.write(msg)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # [FIX] instantiate tqdm-friendly handler
    handler = TqdmLoggingHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RankFilter())
    logger.handlers = [handler]

    # [ADDED] route warnings through logging (keeps tqdm bar clean)
    logging.captureWarnings(True)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    # [MODIFIED] prefer offline-first, write/read run id under checkpoint dir
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    # [ADDED] make wandb write to checkpoint dir and default to offline
    os.environ.setdefault("WANDB_DIR", str(ckpt_dir))

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, mode="offline", dir=str(ckpt_dir))
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            mode="offline",  # [ADDED] offline by default
            dir=str(ckpt_dir),
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


# [ADDED] ---- model summary & param stats helpers ----
def _human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def _sum_params(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad if trainable_only else True))


def _sum_params_direct(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        p.numel()
        for _, p in module._parameters.items()  # noqa: SLF001
        if p is not None and (p.requires_grad if trainable_only else True)
    )


def _build_summary_lines(module: torch.nn.Module, *, name: str = "", depth: int = 0, max_depth: int = 1) -> list[str]:
    cls = module.__class__.__name__
    own = _sum_params_direct(module, trainable_only=False)
    own_t = _sum_params_direct(module, trainable_only=True)
    subtree = _sum_params(module, trainable_only=False)
    subtree_t = _sum_params(module, trainable_only=True)
    indent = "  " * depth
    line = f"{indent}- {name or cls} [{cls}]: own={_human_count(own)} (trainable={_human_count(own_t)}), total={_human_count(subtree)} (trainable={_human_count(subtree_t)})"
    lines = [line]
    if depth < max_depth:
        for child_name, child in module.named_children():
            lines += _build_summary_lines(child, name=child_name, depth=depth + 1, max_depth=max_depth)
    return lines


def _log_model_overview(model: torch.nn.Module, *, is_main: bool, header: str = "Model summary", max_depth: int = 1):
    if not is_main:
        return
    total = _sum_params(model, trainable_only=False)
    trainable = _sum_params(model, trainable_only=True)
    frozen = total - trainable
    logging.info(f"{header}:")
    for ln in _build_summary_lines(model, max_depth=max_depth):
        logging.info(ln)
    logging.info(f"Params total={_human_count(total)} | trainable={_human_count(trainable)} | frozen={_human_count(frozen)}")


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    # [MODIFIED] resolve local_rank/device first and set CUDA device before init_process_group
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        # [ADDED] safer NCCL defaults and diagnostics
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_DEBUG", "WARN")

        # [MODIFIED] try passing device_id for proper rank<->GPU mapping; fallback if unsupported
        try:
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                device_id=(device if torch.cuda.is_available() else None),
            )
        except TypeError:
            # Older PyTorch: no device_id arg
            torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint in a minimal safetensors-only format (plus metadata & optional optimizer)."""
    if not is_main:
        return

    if (global_step % config.save_interval == 0 and global_step > 0) or \
        (global_step == -1) or \
        (global_step == config.num_train_steps - 1):

        latest_ckpt_dir = config.checkpoint_dir / "latest"
        latest_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        # Always save model in safetensors format to latest
        safetensors.torch.save_model(model_to_save, latest_ckpt_dir / "model.safetensors")

        # Save optimizer state if provided (optional)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), latest_ckpt_dir / "optimizer.pt")

        # Save minimal metadata for resume
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": pytime.time(),
        }
        torch.save(metadata, latest_ckpt_dir / "metadata.pt")

        # save norm stats if needed
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(latest_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        logging.info(f"Saved latest checkpoint (safetensors + metadata) -> {latest_ckpt_dir}")

        # Also persist a step-specific checkpoint directory for easier inspection / deterministic resume
        ckpt_dir = config.checkpoint_dir / (f"{global_step}" if global_step > 0 else "pretest_ckpt")
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_model(model_to_save, ckpt_dir / "model.safetensors")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(metadata, ckpt_dir / "metadata.pt")
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        logging.info(f"Saved step checkpoint -> {ckpt_dir}")

        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest safetensors checkpoint and return the global step.

    Minimal behavior:
      - prefer numeric step dirs (max step),
      - fallback to root checkpoint_dir containing model.safetensors,
      - load model.safetensors only (strict=False to tolerate small key diffs),
      - load optimizer.pt if present,
      - read metadata.pt for global_step (fallback to numeric step or 0).
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    # discover step dirs
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if checkpoint_steps:
        latest_step = max(checkpoint_steps)
        ckpt_dir = checkpoint_dir / f"{latest_step}"
    else:
        # fallback to root containing model.safetensors
        if (checkpoint_dir / "model.safetensors").exists():
            latest_step = None
            ckpt_dir = checkpoint_dir
        else:
            raise FileNotFoundError(f"No safetensors checkpoint found in {checkpoint_dir}")

    # Clear memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step if latest_step is not None else 0, "before_loading_checkpoint")

    logging.info(f"Loading model from {ckpt_dir / 'model.safetensors'}")
    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No model.safetensors found at {safetensors_path}")

    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    # Load full model weights (use safetensors loader). Use strict=False tolerance via load_state_dict if needed.
    try:
        safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
        logging.info("Loaded model state from safetensors.")
    except Exception as e:
        # Fallback: try loading into CPU state dict and then load_state_dict (compat mode).
        logging.warning(f"safetensors.load_model failed ({e!s}), attempting fallback load via state_dict.")
        ext_state = safetensors.torch.load_file(safetensors_path, device="cpu")
        cleaned = {}
        for k, v in ext_state.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v
        res = model_to_load.load_state_dict(cleaned, strict=False)
        logging.info(f"Fallback loaded model (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")

    torch.cuda.empty_cache()
    gc.collect()
    log_memory_usage(device, latest_step if latest_step is not None else 0, "after_loading_model")

    # Load optimizer state if available (optional)
    optimizer_path = ckpt_dir / "optimizer.pt"
    if optimizer is not None and optimizer_path.exists():
        opt_state = torch.load(optimizer_path, map_location=device)
        optimizer.load_state_dict(opt_state)
        logging.info("Loaded optimizer state from optimizer.pt")
        del opt_state
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step if latest_step is not None else 0, "after_loading_optimizer")

    # Load metadata for global_step
    metadata_path = ckpt_dir / "metadata.pt"
    global_step = None
    if metadata_path.exists():
        metadata = torch.load(metadata_path, map_location=device)
        global_step = metadata.get("global_step", latest_step if latest_step is not None else 0)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step if latest_step is not None else 0, "after_loading_metadata")
    else:
        global_step = latest_step if latest_step is not None else 0

    logging.info(f"Successfully loaded checkpoint from {ckpt_dir} (step={global_step if global_step is not None else 'root'})")
    return global_step


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    # If no numeric step directories but a root model.safetensors exists, return None (handled by load_checkpoint)
    if not checkpoint_steps and (checkpoint_dir / "model.safetensors").exists():
        return None
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)
    if is_main:
        logging.info(f"Stage: DDP initialized | use_ddp={use_ddp} rank={local_rank} device={device}")

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if not exp_checkpoint_dir.exists():
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
        # Detect numeric-step dirs or a root model.safetensors (both valid resume targets)
        latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
        root_has_model = (exp_checkpoint_dir / "model.safetensors").exists()
        if latest_step is not None or root_has_model:
            resuming = True
            if latest_step is not None:
                logging.info(f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}")
            else:
                logging.info(f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} using root model.safetensors")
        else:
            raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        # [MODIFIED] delete only on main rank and handle FS races
        if is_main:
            try:
                shutil.rmtree(config.checkpoint_dir)
                logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
            except FileNotFoundError:
                logging.info(f"Checkpoint directory already removed: {config.checkpoint_dir}")
            except Exception as e:
                logging.warning(f"Failed to remove checkpoint directory {config.checkpoint_dir}: {e!s}")
        if use_ddp and dist.is_initialized():
            dist.barrier()

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        logging.info("Stage: wandb initialized")

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)
    if is_main:
        try:
            dl_len = len(loader)
        except Exception:
            dl_len = -1
        logging.info(f"Stage: datasets ready | dataloader_len={dl_len} | batch_per_gpu={effective_batch_size}")

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    if is_main:
        logging.info("Stage: model constructed and moved to device")

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
            broadcast_buffers=False,
        )

    # [ADDED] helper: unwrap model if DDP
    def _unwrap(m):
        return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

    # [ADDED] robust full-model loader
    def _load_full_model_weights(mref, path: str, device: torch.device, is_main: bool):
        # 支持目录（默认取 model.safetensors）或文件路径（.safetensors / .pt / .bin）
        p = os.fspath(path)
        if os.path.isdir(p):
            candidate = os.path.join(p, "model.safetensors")
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"full_model_weight_path directory missing model.safetensors: {p}")
            p = candidate

        # 读取 state_dict（兼容常见容器与 DDP 前缀）
        if os.path.splitext(p)[1] == ".safetensors":
            ext_state = safetensors.torch.load_file(p, device="cpu")
        else:
            blob = torch.load(p, map_location="cpu")
            if isinstance(blob, dict):
                for key in ("state_dict", "model", "module", "model_state_dict"):
                    if key in blob and isinstance(blob[key], dict):
                        blob = blob[key]
                        break
            ext_state = blob if isinstance(blob, dict) else {}

        # 去除常见前缀（module.）
        cleaned = {}
        for k, v in ext_state.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v

        # 直接整模加载（strict=False 以兼容少量 shape/缺键差异）
        res = mref.load_state_dict(cleaned, strict=False)
        if is_main:
            logging.info(
                f"Loaded FULL model weights from {path} (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})"
            )

    # [MODIFIED] 权重加载：若提供 full_model_weight_path，则仅一次性加载完整权重；否则沿用分开加载逻辑
    has_full = getattr(config, "full_model_weight_path", None)

    if has_full:
        try:
            mref = _unwrap(model)
            _load_full_model_weights(mref, has_full, device, is_main)
        except Exception as e:
            if is_main:
                logging.warning(f"Failed to load full model weights from {has_full}: {e!s}")
    else:
        # Load weights from weight_loader if specified (for fine-tuning)
        if config.pytorch_weight_path is not None:
            logging.info(f"Loading weights from: {config.pytorch_weight_path}")
            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            try:
                # [UNCHANGED-LOGIC] 过滤掉我们新增模块（预训练不包含）
                file_state = safetensors.torch.load_file(model_path, device=str(device))
                mref = _unwrap(model)
                tgt_keys = set(mref.state_dict().keys())
                filtered = {
                    k: v
                    for k, v in file_state.items()
                    if k in tgt_keys and not (k.startswith("vggt.") or k.startswith("fuser."))
                }
                res = mref.load_state_dict(filtered, strict=False)
                if is_main:
                    logging.info(
                        f"Loaded PyTorch weights (filtered). missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
                    )
            except Exception:
                safetensors.torch.load_model(_unwrap(model), model_path)
                if is_main:
                    logging.info("Loaded PyTorch weights via safetensors.load_model (fallback)")
            logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

        # [UNCHANGED-LOGIC] VGGT 仅加载 aggregator（预训练不含 fuser）
        if getattr(config, "vggt_ckpt_path", None):
            mref = _unwrap(model)
            path = config.vggt_ckpt_path
            raw = torch.load(path, map_location="cpu")
            state = None
            if isinstance(raw, dict):
                for k in ("aggregator", "module.aggregator", "model.aggregator", "vggt", "state_dict", "model_state_dict", "model"):
                    if k in raw and isinstance(raw[k], dict):
                        state = raw[k]
                        break
                if state is None:
                    state = raw
            if not isinstance(state, dict):
                raise ValueError("Unsupported VGGT checkpoint format")
            def _best_prefix_load(module: torch.nn.Module, ext_state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
                tgt_keys = set(module.state_dict().keys())
                counts = {}
                for k in ext_state.keys():
                    parts = k.split(".")
                    for i in range(len(parts) + 1):
                        prefix = ".".join(parts[:i]).rstrip(".")
                        suffix = ".".join(parts[i:])
                        if suffix in tgt_keys:
                            counts[prefix] = counts.get(prefix, 0) + 1
                best_prefix = max(counts, key=counts.get) if counts else ""
                mapped = {}
                if counts:
                    plen = len(best_prefix) + 1 if best_prefix else 0
                    for k, v in ext_state.items():
                        if best_prefix and not k.startswith(best_prefix + "."):
                            continue
                        suffix = k[plen:] if best_prefix else k
                        if suffix in tgt_keys:
                            mapped[suffix] = v
                if not mapped:
                    for pref in ("aggregator.", "module.aggregator.", "module.", "model.", "vggt."):
                        tmp = {}
                        for k, v in ext_state.items():
                            nk = k[len(pref):] if k.startswith(pref) else k
                            if nk in tgt_keys:
                                tmp[nk] = v
                        if tmp:
                            mapped = tmp
                            break
                res = module.load_state_dict(mapped, strict=False)
                return list(res.missing_keys), list(res.unexpected_keys)
            missing, unexpected = _best_prefix_load(mref.vggt, state)
            if is_main:
                logging.info(
                    f"Loaded VGGT Aggregator from {path} (missing={len(missing)}, unexpected={len(unexpected)})"
                )
            
    # [ADDED] optional name-regex allowlist to freeze/unfreeze params
    allowlist = list(getattr(config, "pytorch_trainable_name_regexes", []) or [])
    if allowlist:
        pats = [re.compile(p) for p in allowlist]
        miter = (_unwrap(model)).named_parameters() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.named_parameters()
        for name, p in miter:
            req = any(r.search(name) for r in pats)
            p.requires_grad = bool(req)
        if is_main:
            logging.info(f"Applied trainable allowlist regex to parameters: {allowlist}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    def _iter_trainable_params_for_groups():
        model_for_naming = _unwrap(model)
        return [
            {
                "params": [p for p in model_for_naming.parameters() if p.requires_grad],
                "lr": peak_lr,
                "weight_decay": config.optimizer.weight_decay,
            }
        ]

    optim = torch.optim.AdamW(
        _iter_trainable_params_for_groups(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=0.0,  # handled per-group
    )
    if is_main:
        logging.info(f"Stage: optimizer ready | param_groups={len(optim.param_groups)}")

    # [ADDED] AMP autocast + GradScaler (reduces activation memory)
    amp_dtype = None
    if device.type == "cuda":
        if str(model_cfg.dtype) in ("float16", "fp16") or model_cfg.dtype == torch.float16:
            amp_dtype = torch.float16
        elif str(model_cfg.dtype) in ("bfloat16", "bf16") or model_cfg.dtype == torch.bfloat16:
            amp_dtype = torch.bfloat16
    amp_enabled = (device.type == "cuda") and (amp_dtype is not None)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")
        if is_main:
            logging.info(f"Stage: resume complete | global_step={global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # [ADDED] print model summary & parameter stats before training
    _log_model_overview(_unwrap(model), is_main=is_main, header="Model summary", max_depth=1)

    model.train()
    start_time = pytime.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(
            total=config.num_train_steps,
            initial=global_step,
            desc="Training",
            disable=not is_main,
            dynamic_ncols=True,       # [ADDED]
            smoothing=0.1,            # [ADDED]
            mininterval=0.2,          # [ADDED] throttle redraws
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",  # [ADDED]
        )
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32).to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            # Logging
            if is_main and (global_step % config.log_interval == 0):
                elapsed = pytime.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / max(1, len(infos))
                avg_lr = sum(info["learning_rate"] for info in infos) / max(1, len(infos))

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)

                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / max(1, config.log_interval),
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = pytime.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)
            if is_main and (global_step % max(1, config.save_interval) == 0):
                logging.info(f"Stage: checkpoint saved @ step {global_step}")

            # [MODIFIED] Update progress bar; only set postfix on log intervals
            if pbar is not None:
                pbar.update(1)
                if (global_step % max(1, config.log_interval)) == 0:
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step},
                        refresh=False,
                    )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()