#!/bin/bash

# Fine-tuning script for Connect Four model
# Make sure to update HF_USERNAME with your HuggingFace username

HF_USERNAME="Looyyd"  # Change this to your HuggingFace username
MODEL_NAME="connectfour-qwen2.5-1.5b-instruct"

# Create dataset if it doesn't exist
if [ ! -d "./connectfour_grid_dataset" ]; then
    echo "Creating Connect Four dataset..."
    python create_connectfour_dataset.py
fi

# Option 1: Full fine-tuning (requires more GPU memory)
echo "Starting full fine-tuning..."
accelerate launch ./sft/connectfour_trl.py \
    --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_path "./connectfour_grid_dataset" \
    --output_dir "./connectfour-finetuned" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    # --push_to_hub \
    # --hub_model_id "${HF_USERNAME}/${MODEL_NAME}"

