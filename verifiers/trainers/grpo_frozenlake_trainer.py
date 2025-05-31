import warnings
from typing import Callable, Optional, Union, Any, List, Dict, Sequence
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random
import time
import re

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset
from peft import PeftConfig
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.imports import LLM, SamplingParams
from verifiers.inference.vllm_client import VLLMClient

# monkey patch vllm client
import trl.extras.vllm_client

trl.extras.vllm_client.VLLMClient = VLLMClient

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

if is_wandb_available():
    import wandb

# Add gymnasium import
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

# Grid distribution configuration
DEFAULT_GRID_DISTRIBUTION = {
    2: 0.2,  # 2x2 grids: 33.3%
    3: 0.3,  # 3x3 grids: 33.3%
    4: 0.5,  # 4x4 grids: 33.3%
}


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean(
        (tensor - torch.nanmean(tensor, keepdim=True)) ** 2
    )  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def generate_random_frozenlake_map(size: int, p: float = 0.8) -> List[str]:
    """
    Generate a random FrozenLake map of specified size.

    Args:
        size: Size of the grid (e.g., 2 for 2x2, 3 for 3x3, etc.)
        p: Probability of a frozen tile (vs hole)

    Returns:
        List of strings representing the map
    """
    return generate_random_map(size=size, p=p)


class GRPOFrozenLakeTrainer(GRPOTrainer):
    """
    A GRPO trainer specifically for FrozenLake environment.
    Inherits directly from GRPOTrainer to allow custom modifications.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: Optional[GRPOConfig] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # FrozenLake specific parameters
        is_slippery: bool = False,
        grid_distribution: Optional[Dict[int, float]] = None,
        n_initial_samples: int = 100,
        format_reward_weight: float = 1.0,
        game_reward_weight: float = 10.0,
        max_episode_steps: int = 50,
        frozen_tile_probability: float = 0.8,
        **kwargs,
    ):
        if not args.use_vllm:  # type: ignore
            raise ValueError("vLLM must be enabled for GRPOFrozenLakeTrainer")

        # FrozenLake configuration
        self.is_slippery = is_slippery
        self.grid_distribution = grid_distribution or DEFAULT_GRID_DISTRIBUTION
        self.format_reward_weight = format_reward_weight
        self.game_reward_weight = game_reward_weight
        self.max_episode_steps = max_episode_steps
        self.frozen_tile_probability = frozen_tile_probability

        # Validate grid distribution
        total_prob = sum(self.grid_distribution.values())
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(
                f"Grid distribution probabilities must sum to 1.0, got {total_prob}"
            )

        # Define system prompt
        self.system_prompt = """You are playing a game called frozen lake. 
In this game you move around a grid, and you win when the Agent reaches the Goal. If you fall into a hole, you lose.
You will be given grids by the user, propose the best move.

The Legend is:
A: Agent
S: Start
F: Frozen
H: Hole
G: Goal

Possible moves are:
0: LEFT
1: DOWN
2: RIGHT
3: UP

You should think about your move first using <think></think> tags, then give your final answer.
Put your final answer in \\boxed{}, for example \\boxed{0} for LEFT, \\boxed{1} for DOWN, etc.

Example format:
<think>
I need to analyze the current state and find the best path to the goal while avoiding holes...
</think>

\\boxed{2}"""

        # Store gym environments indexed by a unique ID
        self._gym_envs = {}
        self._next_env_id = 0

        # Multi-turn specific attributes
        self.max_workers = kwargs.pop("max_workers", 10)
        self.sleep_time = kwargs.pop("sleep_time", 0.01)
        self.scale_rewards = kwargs.pop("scale_rewards", True)

        # Create initial dataset (after system_prompt is defined)
        train_dataset = self._create_initial_dataset(n_initial_samples)

        # Define reward functions
        reward_funcs = [self._format_reward_func, self._game_reward_func]

        # Call parent constructor
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=None,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        # Set reward weights after init
        self.reward_weights = torch.tensor(
            [self.format_reward_weight, self.game_reward_weight]
        )

        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            # Config recommended for Qwen 3 thinking, it's probably a good default for thinking tasks
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repetition_penalty=self.repetition_penalty,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def _sample_grid_size(self) -> int:
        """Sample a grid size based on the grid distribution."""
        sizes = list(self.grid_distribution.keys())
        probabilities = list(self.grid_distribution.values())
        return np.random.choice(sizes, p=probabilities)

    def _create_initial_dataset(self, n_samples: int) -> Dataset:
        """Create a dataset with initial FrozenLake states."""
        data = []
        for i in range(n_samples):
            # Sample a grid size
            grid_size = self._sample_grid_size()

            # Generate a random map for this size
            map_desc = generate_random_frozenlake_map(
                size=grid_size, p=self.frozen_tile_probability
            )

            # Get initial state description
            initial_prompt = self._get_initial_state_description_from_map(map_desc)

            # Add system prompt and initial state
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": initial_prompt})

            data.append(
                {
                    "prompt": messages,
                    "map_desc": map_desc,  # Store the map description for later use
                    "grid_size": grid_size,
                }
            )
        return Dataset.from_list(data)

    def _get_initial_state_description_from_map(self, map_desc: List[str]) -> str:
        """Get description of initial state from a map description."""
        # Convert map description to grid
        grid = [list(row) for row in map_desc]

        # Agent starts at position 0 (top-left) which should be 'S'
        return self._state_to_description(0, grid)

    def _get_grid_from_env(self, env: gym.Env) -> List[List[str]]:
        """Extract grid layout from gymnasium environment."""
        desc = env.unwrapped.desc
        # Convert bytes to strings if necessary
        if isinstance(desc[0][0], bytes):
            grid = [[cell.decode("utf-8") for cell in row] for row in desc]
        else:
            grid = [[str(cell) for cell in row] for row in desc]
        return grid

    def _state_to_description(self, state: int, grid: List[List[str]]) -> str:
        """Convert state number to text description."""
        ncol = len(grid[0])
        row = state // ncol
        col = state % ncol

        # Create display grid with agent position
        display_grid = [row[:] for row in grid]  # Deep copy
        display_grid[row][col] = "A"  # Agent symbol

        # Format as string
        grid_str = "Current grid state:\n"
        for grid_row in display_grid:
            grid_str += " ".join(grid_row) + "\n"

        return grid_str

    def _parse_action(self, message: str) -> Optional[int]:
        """Parse action from assistant message in \\boxed{} format."""
        if len(message) == 0:
            return None

        # Look for \boxed{X} pattern where X is a digit 0-3
        import re

        pattern = r"\\boxed\{(\d)\}"
        matches = re.findall(pattern, message)

        if matches:
            # Take the last match in case there are multiple
            digit = matches[-1]
            if digit in "0123":
                return int(digit)

        return None

    def _format_reward_func(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[float]:
        """Reward function that checks if the response format is correct."""
        rewards = []
        # Get episode outcomes from kwargs if available
        episode_outcomes = kwargs.get("episode_outcomes", [None] * len(completions))

        for i, completion in enumerate(completions):
            outcome = episode_outcomes[i]
            if outcome == "invalid_action":
                rewards.append(-0.1)
                continue

            # Check the last assistant message for proper format
            last_assistant_msg = None
            for msg in reversed(completion):
                if msg["role"] == "assistant":
                    last_assistant_msg = msg
                    break

            if last_assistant_msg is None:
                rewards.append(-0.1)
                continue

            content = last_assistant_msg["content"]

            has_thinking = bool(re.search(r"<think>.*?</think>", content, re.DOTALL))

            # Check for valid boxed answer
            action = self._parse_action(content)
            has_valid_answer = action is not None

            # Reward structure:
            # +0.1 if has thinking tags
            # +0.1 if has valid boxed answer
            # 0.0 baseline for valid format
            # -0.1 for invalid action (handled above)

            if has_thinking and has_valid_answer:
                rewards.append(0.2)  # Full credit for perfect format
            elif has_valid_answer:
                rewards.append(0.1)  # Partial credit for answer without thinking
            elif has_thinking:
                rewards.append(0.05)  # Small credit for thinking without valid answer
            else:
                rewards.append(0.0)  # No bonus for poor format

        return rewards

    def _game_reward_func(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[float]:
        """Reward function that returns the game rewards from FrozenLake."""
        rewards = []
        # Get episode outcomes from kwargs if available
        episode_outcomes = kwargs.get("episode_outcomes", [None] * len(completions))

        for outcome in episode_outcomes:
            rewards.append(1.0 if outcome == "goal_reached" else 0.0)

        return rewards

    def generate_multiturn(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM | VLLMClient,
        sampling_params: SamplingParams,
        map_descs: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        """Generate multi-turn FrozenLake episodes."""

        # Initialize states
        states = []
        for i, m in enumerate(prompts):
            env_id = self._next_env_id
            self._next_env_id += 1

            # Create new gym environment with custom map if provided
            if map_descs and i < len(map_descs):
                gym_env = gym.make(
                    "FrozenLake-v1",
                    desc=map_descs[i],
                    is_slippery=self.is_slippery,
                )
            else:
                # Fallback to random 4x4 map
                gym_env = gym.make(
                    "FrozenLake-v1",
                    desc=generate_random_frozenlake_map(
                        4, self.frozen_tile_probability
                    ),
                    is_slippery=self.is_slippery,
                )
            initial_state, _ = gym_env.reset()
            grid = self._get_grid_from_env(gym_env)

            # Store environment
            self._gym_envs[env_id] = {
                "env": gym_env,
                "state": initial_state,
                "grid": grid,
                "done": False,
                "episode_reward": 0.0,
            }

            state = {
                "messages": m,
                "prompt_messages": len(m),
                "prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
                "gym_env_id": env_id,
                "steps": 0,
                "episode_outcome": None,  # Track outcome for reward functions
            }
            states.append(state)

        # Main episode loop
        all_completed = False
        while not all_completed and all(
            s["steps"] < self.max_episode_steps for s in states
        ):
            states = self.step_frozenlake(states, llm, sampling_params)
            all_completed = all(state["completed"] for state in states)

        # Extract results
        completion_messages = [s["messages"][s["prompt_messages"] :] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        episode_outcomes = [s["episode_outcome"] for s in states]

        # Clean up environments
        for env_info in self._gym_envs.values():
            if "env" in env_info:
                env_info["env"].close()
        self._gym_envs.clear()

        return {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
            "episode_outcomes": episode_outcomes,
        }

    def step_frozenlake(
        self,
        states: List[Dict[str, Any]],
        llm: LLM | VLLMClient,
        sampling_params: SamplingParams,
    ) -> List[Dict[str, Any]]:
        """Execute one step of FrozenLake for all active states."""

        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]

        # Get LLM responses
        if isinstance(llm, VLLMClient):
            llm_responses = llm.chat(
                messages_to_step,
                n=1,
                repetition_penalty=sampling_params.repetition_penalty,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                min_p=sampling_params.min_p,
                max_tokens=sampling_params.max_tokens,
                stop=sampling_params.stop,
                include_stop_str_in_output=sampling_params.include_stop_str_in_output,
                skip_special_tokens=sampling_params.skip_special_tokens,
                spaces_between_special_tokens=sampling_params.spaces_between_special_tokens,
            )
            # Convert response format if needed
            from verifiers.envs.multiturn_env import dict_to_chat_response

            llm_responses = dict_to_chat_response(llm_responses).responses
        else:
            llm_responses = llm.chat(
                messages_to_step, sampling_params=sampling_params, use_tqdm=False
            )

        def update_state(j, llm_response):
            # Sleep for rate limiting
            time.sleep(self.sleep_time * random.random())

            state = deepcopy(states[j])
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids

            # Add assistant message
            assistant_msg = {
                "role": "assistant",
                "content": llm_response.outputs[0].text,
            }
            state["messages"].append(assistant_msg)

            # Update token tracking
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # Update completion masks
            state["completion_mask"].extend(
                [0] * env_response_len
            )  # Environment tokens masked
            state["completion_mask"].extend(
                [1] * new_completion_len
            )  # Assistant tokens not masked

            # Update completion ids
            state["completion_ids"] = list(llm_response.prompt_token_ids)
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][
                len(state["prompt_ids"]) :
            ]

            # Parse action and execute gym step
            env_id = state["gym_env_id"]
            env_info = self._gym_envs[env_id]

            action = self._parse_action(assistant_msg["content"])

            if action is None:
                # Invalid action - terminate
                env_info["done"] = True
                state["completed"] = True
                state["episode_outcome"] = "invalid_action"
                # Don't add any message - just mark as completed
            else:
                # Execute action in gym
                gym_env = env_info["env"]
                try:
                    next_state, reward, terminated, truncated, info = gym_env.step(
                        action
                    )
                    done = terminated or truncated

                    env_info["state"] = next_state
                    env_info["done"] = done
                    env_info["episode_reward"] += reward

                    if done:
                        # Episode completed
                        state["completed"] = True
                        # Track outcome without adding messages
                        if reward > 0:
                            state["episode_outcome"] = "goal_reached"
                        else:
                            state["episode_outcome"] = "fell_in_hole"
                        # Don't add any final message
                    else:
                        # Continue episode
                        env_msg = {
                            "role": "user",
                            "content": self._state_to_description(
                                next_state, env_info["grid"]
                            ),
                        }
                        state["messages"].append(env_msg)
                        # Don't update masks here - will be handled in next iteration

                except Exception as e:
                    # Error in gym step
                    state["completed"] = True
                    env_info["done"] = True
                    state["episode_outcome"] = "error"
                    raise RuntimeError(
                        f"Error executing action in gym environment: {str(e)}"
                    )

            # Increment step counter
            state["steps"] += 1

            # Truncate if too long
            if len(state["completion_ids"]) > sampling_params.max_tokens:
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][
                    : sampling_params.max_tokens
                ]
                state["completion_mask"] = state["completion_mask"][
                    : len(state["completion_ids"])
                ]

            # Ensure mask and ids have same length
            min_len = min(len(state["completion_mask"]), len(state["completion_ids"]))
            state["completion_mask"] = state["completion_mask"][:min_len]
            state["completion_ids"] = state["completion_ids"][:min_len]

            return j, state

        # Execute updates in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                executor.map(
                    lambda args: update_state(*args),
                    [(j, llm_responses[i]) for i, j in enumerate(live_indices)],
                )
            )

        for j, state in results:
            states[j] = state

        return states

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Override to use FrozenLake-specific generation."""

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather prompts for multi-turn generation
        all_prompts = gather_object(prompts)
        # Extract map descriptions if available
        map_descs = (
            [x.get("map_desc") for x in inputs] if "map_desc" in inputs[0] else None
        )
        all_map_descs = gather_object(map_descs) if map_descs else None

        if self.accelerator.is_main_process:
            env_result = self.generate_multiturn(
                prompts=all_prompts,
                llm=self.vllm_client,
                sampling_params=self.sampling_params,
                map_descs=all_map_descs,
            )
            completion_ids = env_result["ids"]
            completion_messages = env_result["messages"]
            completion_mask = env_result["mask"]
            episode_outcomes = env_result.get(
                "episode_outcomes", [None] * len(all_prompts)
            )
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
            episode_outcomes = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        episode_outcomes = broadcast_object_list(episode_outcomes, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        episode_outcomes = episode_outcomes[process_slice]

        # Pad completions
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
        )

        completion_mask = [
            torch.tensor(mask, device=device) for mask in completion_mask
        ]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)

        # Compute logps
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

        # Compute rewards
        completions = completion_messages
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, reward_func in enumerate(self.reward_funcs):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            # Add episode outcomes to reward kwargs
            reward_kwargs["episode_outcomes"] = episode_outcomes
            output_reward_func = reward_func(
                prompts=prompts, completions=completions, **reward_kwargs
            )

            output_reward_func = [
                reward if reward is not None else torch.nan
                for reward in output_reward_func
            ]
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        rewards_per_func = gather(rewards_per_func)

        # Apply weights
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards

        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice for local process
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)

        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        [str(prompts_to_log[0][-1]["content"])],
                        [completions_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def __del__(self):
        """Clean up gym environments when trainer is destroyed."""
        for env_info in self._gym_envs.values():
            if "env" in env_info:
                env_info["env"].close()
