import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import PartialState
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="connectfour_grid_dataset",
        help="Path to the dataset created by the generation script",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./connectfour-qwen-finetuned",
        help="Output directory for the model",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="Looyyd/connectfour-qwen2.5-1.5b",
        help="HuggingFace Hub model ID for uploading",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the model to HuggingFace Hub after training",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)

    # Model configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Define formatting function after tokenizer is loaded
    def format_chat_template(example):
        """Format the messages into the chat template."""
        # Apply the chat template to the messages
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    # Model loading configuration
    if args.use_lora:
        # Use 4-bit quantization with LoRA for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Full fine-tuning
        print(f"Loading model for full fine-tuning...")
        # For distributed training with DDP, follow the documentation guidance
        device_string = PartialState().process_index
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device_string},
            trust_remote_code=True,
        )

    # SFT configuration using SFTConfig instead of TrainingArguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for DDP
        optim="adamw_torch",
        learning_rate=2e-4 if args.use_lora else 5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        report_to=["wandb"],
        ddp_find_unused_parameters=False,
        group_by_length=True,
        dataloader_num_workers=4,
        # SFT-specific parameters
        max_length=2048,
        dataset_text_field="text",
        packing=False,
        # For Qwen models with predefined chat template
        eos_token="<|im_end|>",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=format_chat_template,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model()

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # Merge LoRA weights if using LoRA
    if args.use_lora:
        print("Merging LoRA weights...")
        from peft import AutoPeftModelForCausalLM

        # Load and merge
        merged_model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        merged_model = merged_model.merge_and_unload()

        # Save merged model
        merged_output_dir = f"{args.output_dir}-merged"
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        print(f"Merged model saved to {merged_output_dir}")

        # Update output dir for pushing
        if args.push_to_hub:
            args.output_dir = merged_output_dir

    # Push to hub if requested
    if args.push_to_hub:
        print(f"Pushing model to HuggingFace Hub as {args.hub_model_id}...")
        if args.use_lora:
            # Push the merged model
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_folder(
                folder_path=args.output_dir,
                repo_id=args.hub_model_id,
                repo_type="model",
                commit_message="Upload fine-tuned Connect Four Qwen model",
            )
        else:
            trainer.push_to_hub(
                commit_message="Upload fine-tuned Connect Four Qwen model"
            )
        print(
            f"Model successfully pushed to: https://huggingface.co/{args.hub_model_id}"
        )

    print("Training completed!")


if __name__ == "__main__":
    main()
