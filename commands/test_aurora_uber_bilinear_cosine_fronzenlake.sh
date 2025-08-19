#!/bin/bash

source .env
# export HF_DATASETS_OFFLINE=1
# export OLMO_SHARED_FS=1
# export OMP_NUM_THREADS=8
# export LOG_FILTER_TYPE="rank0_only"
# export TORCH_LOGS_RANK0="recompiles,graph_breaks"
# export OLMO_NUM_THREADS_ENV_VAR=8
# export NCCL_IB_HCA="^=mlx5_bond_0"

# # Define distributed parameters
# export RANK=${RANK:-0}
# export ADDR=${ADDR:-127.0.0.1}
# export PORT=${PORT:-29401}
# export RDZV_ID=$(date +%s%N)

# get date
export DATE=$(date +%Y-%m-%d_%H-%M-%S)

# Define experiment name
export EXP_NAME="lowLR-frozenlake-molmo-7b-qwen2-siglip2-finetune-uber-cosine-bilinear-test" 
# add date to experiment name
EXP_NAME="${DATE}-${EXP_NAME}"

# Check NVIDIA status
nvidia-smi
# Run training

# check if the checkpoints directory for the experiment exists, if yes then delete it
if [ -d "${OUTPUT_DIR}/${EXP_NAME}" ]; then
  echo "Removing existing checkpoints directory: ${OUTPUT_DIR}/${EXP_NAME}"
  rm -rf ${OUTPUT_DIR}/${EXP_NAME}
else
  echo "No existing checkpoints directory found for ${EXP_NAME}."
fi

AZFUSE_USE_FUSE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=23502 \
  --nnodes=1 \
  --nproc-per-node=4 \
  launch_scripts/train_multitask_model.py \
  fronzenlake_debug \
  pretrained/step30000-unsharded/ \
  --wandb.name="${EXP_NAME}" \
  --wandb.entity=${WANDB_TEAM} \
  --wandb.project=${WANDB_PROJECT} \
  --save_folder=${OUTPUT_DIR}/${EXP_NAME} \
  --save_overwrite \
  --image_generation_loss_type="cosine" \
  --per_image_output_tokens=64 \
  --vision_head_type="Linear" \
  --duration=2000 \
  --device_train_batch_size=2 \
  --global_batch_size=32


# # check if the predictions directory exists, if not create it
# if [ ! -d "./predictions/${EXP_NAME}" ]; then
#   mkdir -p ./predictions/${EXP_NAME}
# fi
# # move the _default directory to the predictions directory
# if [ -d "./_default" ]; then
#   mv ./_default ./predictions/${EXP_NAME}
# else
#   echo "No _default directory found to move."
# fi
