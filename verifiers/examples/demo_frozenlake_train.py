from trl import GRPOConfig
from transformers import AutoTokenizer

# Import our new trainer
from verifiers.trainers.grpo_frozenlake_trainer import GRPOFrozenLakeTrainer


"""
2-GPU training (single node, 1 training + 1 inference)

CUDA_VISIBLE_DEVICES=0 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-1.5B-Instruct' --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/demo_frozenlake_train.py
---
4-GPU training (single node, 2 training + 2 inference)

CUDA_VISIBLE_DEVICES=0,1 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-1.5B-Instruct' --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml verifiers/examples/demo_frozenlake_train.py
---
8-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model  'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num-processes 4 --config-file configs/zero3.yaml verifiers/examples/demo_frozenlake_train.py
"""

# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Configuration options
IS_SLIPPERY = False  # Set to True for more challenging environment
BATCH_SIZE = 16  # Reduced from 16 for initial testing
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
    # Trying to save vram with 8bit optimizer, TODO: remove if unstable training
    optim="paged_adamw_8bit",
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    num_train_epochs=1,
    # Config recommended for Qwen 3 thinking, it's probably a good default for thinking tasks
    # TODO: this is also defined in the sampling_params in the trainer, need to unify them, not sure which is actually used
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    max_steps=1000,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=1,
    # KL penalty coefficient, default is 0.04, other demos in this repo use lower kl,
    # some people online used smaller kl also https://x.com/abacaj/status/1886497011618197748
    # TODO: figure out if this works well
    beta=0.001,
    max_prompt_length=512,
    # TODO: need to increase this for multi step reasoning. or implement a method to contract the prompt length.
    max_completion_length=2048,
    per_device_train_batch_size=BATCH_SIZE,
    num_generations=16,
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
    # Dr GRPO
    scale_rewards=False,
    loss_type="dr_grpo",
    # DAPO paper, epsilon_high=0.28 seems the most useful contribution
    epsilon_high=0.28,
)

# Create and run trainer
trainer = GRPOFrozenLakeTrainer(
    model=model_name,
    args=training_args,
    processing_class=tokenizer,
    is_slippery=IS_SLIPPERY,
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
