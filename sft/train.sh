#!/bin/bash

# Login to HuggingFace (required for pushing model)
# Uncomment and run this if you haven't logged in yet
# huggingface-cli login

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4

HF_USERNAME="Looyyd"

# Launch training with accelerate
echo "Starting training on 4 GPUs..."

accelerate launch --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    train_connectfour.py \
    --dataset_path connectfour_grid_dataset \
    --output_dir ./connectfour-qwen-lora \
    --hub_model_id "$HF_USERNAME/connectfour-qwen2.5-1.5b-full" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --push_to_hub


echo "Training completed!"
