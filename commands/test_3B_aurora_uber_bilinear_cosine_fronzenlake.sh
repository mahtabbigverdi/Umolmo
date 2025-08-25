#!/bin/bash

source .env_cloud
# get date
export DATE=$(date +%Y-%m-%d_%H-%M-%S)

# Define experiment name
export EXP_NAME="lowLR-frozenlake-molmo-3b-qwen2-siglip2-finetune-uber-cosine-bilinear-test-4GPU" 
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

AZFUSE_USE_FUSE=0  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=23501 \
  --nnodes=1 \
  --nproc-per-node=4 \
  launch_scripts/train_multitask_model.py \
  frozenlake_debug \
  pretrained/3B-step30000-unsharded/ \
  --wandb.name="${EXP_NAME}" \
  --wandb.entity=${WANDB_TEAM} \
  --wandb.project=${WANDB_PROJECT} \
  --save_folder=${OUTPUT_DIR}/${EXP_NAME} \
  --save_overwrite \
  --image_generation_loss_type="cosine" \
  --per_image_output_tokens=64 \
  --vision_head_type="Linear" \
  --duration=2000 \
  --device_train_batch_size=4 \
  --global_batch_size=32
