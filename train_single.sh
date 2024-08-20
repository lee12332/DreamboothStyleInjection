#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_NAME="stabilityai/stable-diffusion-2-1"
BASE_INSTANCE_DIR="path-to-your-dataset"
OUTPUT_DIR_PREFIX="./ckpt/"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=500
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"     
LR_WARMUP_STEPS=400
MAX_TRAIN_STEPS=800
SEED=2024
GPU_COUNT=1
CUDA_VISIBLE_DEVICES=0
ITEM_NUM=100

current_folder_number=0

INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
PROMPT=$(printf "style_%02d" $current_folder_number)

COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt=$PROMPT \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler=$LR_SCHEDULER \
    --lr_warmup_steps=$LR_WARMUP_STEPS \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --seed=$SEED \
    --item_num=$ITEM_NUM" 

eval $COMMAND &
sleep 2