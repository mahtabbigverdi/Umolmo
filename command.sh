#!/bin/bash

# Set environment variables manually
export MOLMO_DATA_DIR=/gscratch/krishna/mahtab/Umolmo/Data
export HF_HOME=/gscratch/krishna/mahtab/Umolmo/huggingface
export PYTHONPATH=/gscratch/krishna/mahtab/Umolmo:$PYTHONPATH
# export NCCL_TIMEOUT_MINUTES=30
# export HF_DATASETS_OFFLINE=1
# export OLMO_SHARED_FS=1
# export OMP_NUM_THREADS=8
# export LOG_FILTER_TYPE="rank0_only"
# export TORCH_LOGS_RANK0="recompiles,graph_breaks"
# export OLMO_NUM_THREADS_ENV_VAR=8
# export NCCL_IB_HCA="^=mlx5_bond_0"
# export NCCL_DEBUG=INFO
# export ALLENACT_DEBUG_VST_TIMEOUT=5000
# export ALLENACT_DEBUG=true
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_TIMEOUT=36000000

# # Define distributed parameters
# export RANK=${RANK:-0}
# export ADDR=${ADDR:-127.0.0.1}
# export PORT=${PORT:-29401}
# export RDZV_ID=$(date +%s%N)

# Define experiment name
export EXP_NAME="molmo-7b-qwen2-siglip2-finetune"

# Check NVIDIA status
nvidia-smi
rm -rf /gscratch/krishna/mahtab/Umolmo/checkpoints/${EXP_NAME}
rm -rf /gscratch/krishna/mahtab/Umolmo/debug_run

# Run training
HF_ACCESS_TOKEN=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 \
WANDB_API_KEY=42e8013627067866a191055811b0107b24891809 \
# torchrun \
#   --master_port=23501 \
#   --nnodes=1 \
#   --nproc-per-node=1 \
#   launch_scripts/train_multitask_model.py \
#   debug \
#   debug \
#   # --wandb.name=${EXP_NAME} \
#   # --wandb.entity=allenai-team1 \
#   # --wandb.project=mmseek \
#   # --save_folder=/gscratch/krishna/mahtab/Umolmo/checkpoints/${EXP_NAME} \
#   # --save_overwrite 


torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  launch_scripts/train_multitask_model.py \
  smallmahtab \
  /mmfs1/gscratch/krishna/mahtab/Umolmo/pretrained/step22347-unsharded \
  --wandb.name="${EXP_NAME}" \
  --wandb.entity=allenai-team1 \
  --wandb.project=mmseek \
  --save_folder=/mmfs1/gscratch/krishna/mahtab/Umolmo/checkpoints/${EXP_NAME} \
  --save_overwrite 