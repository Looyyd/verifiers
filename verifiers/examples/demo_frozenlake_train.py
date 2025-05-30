from trl import GRPOConfig
from transformers import AutoTokenizer

# Import our new trainer
from verifiers.trainers.grpo_frozenlake_trainer import GRPOFrozenLakeTrainer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

"""
2-GPU training (single node, 1 training + 1 inference)

CUDA_VISIBLE_DEVICES=0 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-1.5B-Instruct' --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/demo_frozenlake_train.py
---
4-GPU training (single node, 2 training + 2 inference)

CUDA_VISIBLE_DEVICES=0,1 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-1.5B-Instruct' --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml verifiers/examples/demo_frozenlake_train.py
"""

# Configuration options
IS_SLIPPERY = False  # Set to True for more challenging environment
BATCH_SIZE = 4  # Reduced from 16 for initial testing
N_INITIAL_SAMPLES = 1000  # Number of initial states in dataset
FORMAT_REWARD_WEIGHT = 1.0  # Weight for format correctness
GAME_REWARD_WEIGHT = 10.0  # Weight for reaching the goal
MAX_EPISODE_STEPS = 50  # Maximum steps per episode

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

run_name = (
    f"demo-frozenlake-{'slippery' if IS_SLIPPERY else 'normal'}-grpo_"
    + model_name.split("/")[-1].lower()
)

training_args = GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    num_train_epochs=1,
    temperature=1.0,
    max_steps=100,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=1,
    beta=0.1,  # KL penalty coefficient
    max_prompt_length=512,
    max_completion_length=1536,
    per_device_train_batch_size=BATCH_SIZE,
    num_generations=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
)

# Create and run trainer
trainer = GRPOFrozenLakeTrainer(
    model=model_name,
    args=training_args,
    processing_class=tokenizer,
    is_slippery=IS_SLIPPERY,
    map_name="4x4",
    n_initial_samples=N_INITIAL_SAMPLES,
    format_reward_weight=FORMAT_REWARD_WEIGHT,
    game_reward_weight=GAME_REWARD_WEIGHT,
    max_episode_steps=MAX_EPISODE_STEPS,
)

print(f"Starting FrozenLake GRPO training (slippery={IS_SLIPPERY})")
print(f"Model: {model_name}")
print(f"Dataset size: {N_INITIAL_SAMPLES} initial states")
print(
    f"Batch size: {BATCH_SIZE}, Generations per prompt: {training_args.num_generations}"
)
print(f"Reward weights - Format: {FORMAT_REWARD_WEIGHT}, Game: {GAME_REWARD_WEIGHT}")

trainer.train()

# Save the final model
trainer.save_model(f"outputs/{run_name}/final_model")
