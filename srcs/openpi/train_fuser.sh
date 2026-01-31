export HF_ENDPOINT=https://hf-mirror.com
export LEROBOT_SKIP_CODEBASE_VERSION_CHECK=1
export OPENPI_COMPILE_SAMPLE=1

uv run scripts/compute_norm_stats.py --config-name pi05_vggt_libero_pytorch
uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    scripts/train_fuser.py pi05_vggt_libero_pytorch --exp_name pi05_vggt_sft_pytorch_bf16