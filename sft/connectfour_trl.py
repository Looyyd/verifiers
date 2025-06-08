"""
Fine-tune a model on the Connect Four grid visualization dataset.

Usage:
python finetune_connectfour.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_path ./connectfour_grid_dataset \
    --output_dir ./connectfour-finetuned \
    --hub_model_id Looyyd/connectfour-qwen2-0.5b
"""

import os
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType
import torch


def main():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./connectfour_grid_dataset",
        help="Path to the Connect Four dataset",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./connectfour-finetuned",
        help="The output directory where the model predictions and checkpoints will be written",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/CPU for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="The initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA for efficient fine-tuning",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Whether to use 4-bit quantization",
    )

    # HuggingFace Hub arguments
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository on HuggingFace Hub",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Split dataset if needed (use 90% for training, 10% for validation)
    if "train" not in dataset.column_names:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation examples: {len(eval_dataset)}")

    # Model configuration
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }

    # 4-bit quantization config
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    # Load model and tokenizer
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, **model_kwargs
    )

    # LoRA configuration
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    config = SFTConfig(
        completion_only_loss=True,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to=["wandb"],
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()

    # Push to hub if requested
    if args.push_to_hub:
        print(f"Pushing model to HuggingFace Hub: {args.hub_model_id}")
        trainer.push_to_hub()

    print("Training complete!")


if __name__ == "__main__":
    main()
