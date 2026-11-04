#!/bin/bash
#
# GPUT device ID to use
DEVICES="0"
EXP_NAME="CL-UAP"

# ==================================================
MODEL="resnet152"
TARGET="None"  # set to class index for targeted attack, e.g., 379
#deit_small_pert16_224.fb_in1k

# Create top-level log directory
TOP_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
TOP_LOG_DIR="./logs/${TOP_TIMESTAMP}_${EXP_NAME}"
mkdir -p "$TOP_LOG_DIR"

# Backup current script to log directory
SCRIPT_PATH="$0"
cp "$SCRIPT_PATH" "${TOP_LOG_DIR}/run_attack_${EXP_NAME}.sh"
echo "Script backed up to ${TOP_LOG_DIR}/run_attack_${EXP_NAME}.sh"

LOG_DIR="${TOP_LOG_DIR}/"
mkdir -p "$LOG_DIR"

# Run attack
CUDA_VISIBLE_DEVICES=$DEVICES \
    python attack.py \
    --log_dir $LOG_DIR \
    --train_models $MODEL \
    --val_models $MODEL \
    --attack_method growing_pert \
    --start_pert_size 28\
    --final_pert_size 224 \
    --target_label $TARGET \
    --cross_entropy_loss_weight 1 \
    --target_cross_entropy_loss_weight 0 \
    --loss_clamp 12 \
    --batch_size 64 \
    --alpha 7 \
    --nb_epoch 20 \
    --random_seed 0
