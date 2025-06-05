from trl import GRPOConfig
from transformers import AutoTokenizer
import bitsandbytes as bnb


# Import our new trainer
from verifiers.trainers.grpo_connectfour_trainer import GRPOConnectFourTrainer


# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Configuration options
BATCH_SIZE = 4
NUM_GENERATIONS = BATCH_SIZE
N_INITIAL_SAMPLES = 1000  # Number of initial states in dataset
FORMAT_REWARD_WEIGHT = 1.0  # Weight for format correctness
GAME_REWARD_WEIGHT = 10.0  # Weight for reaching the goal
MAX_EPISODE_STEPS = 25  # Maximum steps per episode, should always be enough for 4x4 env

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

run_name = f"demo-connectfour-grpo_" + model_name.split("/")[-1].lower()

training_args = GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    num_train_epochs=1,
    # Config recommended for Qwen 3 thinking, it's probably a good default for thinking tasks
    # TODO: i think it's maybe not good, because reduces variety during training, so less exploration
    temperature=1,
    # top_p=0.95,
    # top_k=20,
    min_p=0.0,
    max_steps=1000,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=1,
    # KL penalty coefficient, default is 0.04, other demos in this repo use lower kl,
    # some people online used smaller kl also https://x.com/abacaj/status/1886497011618197748
    beta=0.001,
    max_prompt_length=512,
    max_completion_length=2048,
    per_device_train_batch_size=BATCH_SIZE,
    num_generations=NUM_GENERATIONS,
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
trainer = GRPOConnectFourTrainer(
    model=model_name,
    args=training_args,
    processing_class=tokenizer,
    n_initial_samples=N_INITIAL_SAMPLES,
    format_reward_weight=FORMAT_REWARD_WEIGHT,
    game_reward_weight=GAME_REWARD_WEIGHT,
    max_episode_steps=MAX_EPISODE_STEPS,
    compression_threshold=0.75,
    # Trying to save vram with 8bit optimizer, TODO: remove if unstable training
    optimizers=(
        bnb.optim.Adam8bit,
        None,
    ),
)

print(f"Starting ConnectFour GRPO training")
print(f"Model: {model_name}")
print(f"Dataset size: {N_INITIAL_SAMPLES} initial states")
print(
    f"Batch size: {BATCH_SIZE}, Generations per prompt: {training_args.num_generations}"
)
print(f"Reward weights - Format: {FORMAT_REWARD_WEIGHT}, Game: {GAME_REWARD_WEIGHT}")

trainer.train()

# Save the final model
trainer.save_model(f"outputs/{run_name}/final_model")
