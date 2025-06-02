import warnings
from typing import Callable, Optional, Union, Any, List, Dict, Sequence
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random
import time
import re
import pandas as pd

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
from verifiers.envs.multiturn_env import dict_to_chat_response


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

DEBUG = True

# Grid distribution configuration
DEFAULT_GRID_DISTRIBUTION = {
    2: 0.2,  # 2x2 grids: 33.3%
    3: 0.3,  # 3x3 grids: 33.3%
    4: 0.5,  # 4x4 grids: 33.3%
}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


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

    This trainer extends GRPOTrainer with:
    - Multi-turn FrozenLake environment interaction
    - Custom reward functions for format compliance and game success
    - Optional context compression for long episodes

    Context Compression:
    When enabled, the trainer will prompt the model to summarize the conversation
    when it reaches a token threshold. This allows for training on longer episodes
    without exceeding context limits. The model learns to generate effective
    summaries that preserve important information.

    Example usage with context compression:

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
        # Context compression parameters
        use_context_compression: bool = True,
        compression_threshold: float = 0.75,
        compression_prompt_template: str = (
            "This conversation is getting long. Sum up this conversation so far and the summary "
            "will be given to your next instance to continue the task. You MUST use <think> tags "
            "to organize your thoughts. The content after </think> will be given to your next instance. "
            "If no <think> tags are used, no summary will be given."
        ),
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

        # Context compression configuration
        self.use_context_compression = use_context_compression
        self.compression_threshold = compression_threshold
        self.compression_prompt_template = compression_prompt_template

        # Validate compression parameters
        if self.use_context_compression:
            if not (0.0 < self.compression_threshold <= 1.0):
                raise ValueError(
                    f"compression_threshold must be between 0 and 1, got {self.compression_threshold}"
                )
            if not self.compression_prompt_template:
                raise ValueError(
                    "compression_prompt_template cannot be empty when using context compression"
                )

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

        # Initialize parent class attributes we use in _compute_loss
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )

        # Initialize attributes for _prepare_inputs override
        self._step = 0
        self._buffered_inputs = None

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
        compression_info = kwargs.get("compression_info", [{}] * len(completions))

        for i, completion in enumerate(completions):
            outcome = episode_outcomes[i]
            comp_info = compression_info[i]

            if outcome == "invalid_action":
                rewards.append(-0.1)
                continue

            # Check the last assistant message for proper format
            last_assistant_msg = None
            for msg in reversed(completion):
                if msg["role"] == "assistant":
                    last_assistant_msg = msg
                    break

            # We can only check the last one, because if 1 message is incorrect we end the episode
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
            # Additional penalty for compression without think tags

            base_reward = 0.0
            if has_thinking and has_valid_answer:
                base_reward = 0.2  # Full credit for perfect format
            elif has_valid_answer:
                base_reward = 0.1  # Partial credit for answer without thinking
            elif has_thinking:
                base_reward = 0.05  # Small credit for thinking without valid answer
            else:
                base_reward = 0.0  # No bonus for poor format

            # TODO: Add penalty if messages contain think tags but no compression was done
            # Apply penalty if this was a compression response without think tags
            # (This information would need to be passed through the episode data)
            # For now, we'll skip this as it would require modifying the data flow

            rewards.append(base_reward)

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
        compression_info = kwargs.get("compression_info", [{}] * len(completions))

        for outcome, comp_info in zip(episode_outcomes, compression_info):
            base_reward = 1.0 if outcome == "goal_reached" else 0.0

            # Optional: small penalty for needing compression (can be disabled by setting to 1.0)
            compression_penalty = 1.0  # 5% penalty for needing compression
            if comp_info.get("needed_compression", False) and compression_penalty < 1.0:
                base_reward *= compression_penalty

            rewards.append(base_reward)

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
                # Context compression tracking
                "is_compressing": False,
                "has_been_compressed": False,
                "compression_count": 0,
                # Track conversation segments for proper loss computation
                "conversation_segments": [],
                "current_segment_start": 0,  # Track where current segment starts in completion_ids
                "initial_prompt_ids": None,  # Store the initial prompt_ids for segment creation
                "history_for_logging": deepcopy(m),  # Keep full history for logging
            }
            states.append(state)

        # Main episode loop
        all_completed = False
        while not all_completed and all(
            s["steps"] < self.max_episode_steps for s in states
        ):
            states = self.step_frozenlake(states, llm, sampling_params)
            all_completed = all(state["completed"] for state in states)

        # Ensure we capture final segments before cleanup
        if self.use_context_compression:
            for state in states:
                # Always capture the final segment if there are any completion tokens
                if state["current_segment_start"] < len(state["completion_ids"]):
                    # Add the final segment
                    segment_completion_ids = state["completion_ids"][
                        state["current_segment_start"] :
                    ]
                    segment_completion_mask = state["completion_mask"][
                        state["current_segment_start"] :
                    ]

                    # Use the current prompt_ids for this segment
                    segment_prompt_ids = (
                        state["prompt_ids"]
                        if state["prompt_ids"]
                        else state.get("initial_prompt_ids", [])
                    )

                    if (
                        segment_completion_ids and segment_prompt_ids
                    ):  # Only add if we have both
                        state["conversation_segments"].append(
                            {
                                "prompt_ids": segment_prompt_ids,
                                "completion_ids": segment_completion_ids,
                                "completion_mask": segment_completion_mask,
                            }
                        )
                    else:
                        # This shouldn't happen
                        print(f"WARNING: Missing data for segment creation")
                        print(
                            f"  segment_completion_ids: {len(segment_completion_ids) if segment_completion_ids else 'None'}"
                        )
                        print(
                            f"  segment_prompt_ids: {len(segment_prompt_ids) if segment_prompt_ids else 'None'}"
                        )
                        print(
                            f"  current_segment_start: {state['current_segment_start']}"
                        )
                        print(f"  len(completion_ids): {len(state['completion_ids'])}")
                else:
                    # Debug why we might not have segments
                    if len(state["completion_ids"]) == 0:
                        print(
                            f"WARNING: Episode completed with no completion_ids at all"
                        )
                        print(
                            f"  episode_outcome: {state.get('episode_outcome', 'Unknown')}"
                        )
                        print(f"  steps: {state.get('steps', 0)}")
                        print(f"  messages: {len(state.get('messages', []))}")

        # Extract results with conversation segments
        if self.use_context_compression:
            # Return arrays of segments for each episode
            all_prompt_ids = []
            all_prompt_masks = []
            all_completion_ids = []
            all_completion_masks = []

            for state in states:
                # Extract segments
                episode_prompt_ids = []
                episode_prompt_masks = []
                episode_completion_ids = []
                episode_completion_masks = []

                for segment in state["conversation_segments"]:
                    episode_prompt_ids.append(segment["prompt_ids"])
                    episode_completion_ids.append(segment["completion_ids"])
                    episode_completion_masks.append(segment["completion_mask"])
                    # Create prompt mask of all 1s
                    episode_prompt_masks.append([1] * len(segment["prompt_ids"]))

                # Ensure we have at least one segment per episode
                if not episode_prompt_ids:
                    # This should not happen if the logic above is correct
                    raise RuntimeError(
                        f"Episode completed with no segments captured. "
                        f"completion_ids length: {len(state['completion_ids'])}, "
                        f"current_segment_start: {state['current_segment_start']}"
                    )

                all_prompt_ids.append(episode_prompt_ids)
                all_prompt_masks.append(episode_prompt_masks)
                all_completion_ids.append(episode_completion_ids)
                all_completion_masks.append(episode_completion_masks)

            completion_messages = [
                s["messages"][s["prompt_messages"] :] for s in states
            ]
            history_for_logging = [s["history_for_logging"] for s in states]
            episode_outcomes = [s["episode_outcome"] for s in states]
            compression_info = [
                {
                    "needed_compression": s.get("has_been_compressed", False),
                    "compression_count": s.get("compression_count", 0),
                }
                for s in states
            ]

            # Clean up environments
            for env_info in self._gym_envs.values():
                if "env" in env_info:
                    env_info["env"].close()
            self._gym_envs.clear()

            return {
                "prompt_ids": all_prompt_ids,
                "prompt_masks": all_prompt_masks,
                "completion_ids": all_completion_ids,
                "completion_masks": all_completion_masks,
                "messages": completion_messages,
                "history_for_logging": history_for_logging,
                "episode_outcomes": episode_outcomes,
                "compression_info": compression_info,
                "use_segments": True,  # Flag to indicate segmented data
            }
        else:
            # Original non-compression path
            completion_ids = [s["completion_ids"] for s in states]
            completion_mask = [s["completion_mask"] for s in states]
            episode_outcomes = [s["episode_outcome"] for s in states]
            completion_messages = [
                s["messages"][s["prompt_messages"] :] for s in states
            ]
            history_for_logging = [s["history_for_logging"] for s in states]

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
            llm_responses = dict_to_chat_response(llm_responses).responses
        else:
            llm_responses = llm.chat(
                messages_to_step, sampling_params=sampling_params, use_tqdm=False
            )

        def update_state(j, llm_response):
            # Sleep for rate limiting
            time.sleep(self.sleep_time * random.random())

            state = deepcopy(states[j])

            # Initialize prompt_ids on first call
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids
                # Store initial prompt_ids if not already stored
                if state["initial_prompt_ids"] is None:
                    state["initial_prompt_ids"] = list(llm_response.prompt_token_ids)

            # Add assistant message
            assistant_msg = {
                "role": "assistant",
                "content": llm_response.outputs[0].text,
            }
            state["messages"].append(assistant_msg)
            state["history_for_logging"].append(
                assistant_msg
            )  # Also add to logging history

            # Update token tracking - APPEND, don't overwrite
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # Extend completion masks
            state["completion_mask"].extend(
                [0] * env_response_len
            )  # Environment tokens masked
            state["completion_mask"].extend(
                [1] * new_completion_len
            )  # Assistant tokens not masked

            # Extend completion ids - don't overwrite!
            new_tokens = list(llm_response.prompt_token_ids)[total_prev_len:]
            new_tokens.extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"].extend(new_tokens)

            # Handle compression response
            if state.get("is_compressing", False):
                # Extract summary (after </think> if present)
                summary_text = assistant_msg["content"]

                # Check if the model used think tags
                has_think_tags = "</think>" in summary_text

                if has_think_tags:
                    # Extract content after </think>
                    summary_text = summary_text.split("</think>", 1)[1].strip()
                else:
                    # Empty summary if no think tags
                    # Optionally, we could add a small penalty for not using think tags
                    # This could be tracked and used in the format reward function
                    state["compression_missing_think"] = True
                    summary_text = ""

                # Ensure we have some summary text
                if not summary_text.strip():
                    # Fallback to a minimal summary if empty
                    summary_text = "Previous conversation summary unavailable."

                # Save current segment before compression
                segment_completion_ids = state["completion_ids"][
                    state["current_segment_start"] :
                ]
                segment_completion_mask = state["completion_mask"][
                    state["current_segment_start"] :
                ]

                # Use the current prompt_ids for this segment
                segment_prompt_ids = (
                    state["prompt_ids"]
                    if state["prompt_ids"]
                    else state.get("initial_prompt_ids", [])
                )

                state["conversation_segments"].append(
                    {
                        "prompt_ids": segment_prompt_ids,
                        "completion_ids": segment_completion_ids,
                        "completion_mask": segment_completion_mask,
                    }
                )

                # Get current game state
                env_info = self._gym_envs[state["gym_env_id"]]
                current_state_desc = self._state_to_description(
                    env_info["state"], env_info["grid"]
                )

                # Build new compressed user message
                compressed_user_msg = (
                    f"{current_state_desc}\n\n"
                    f"Here is a message from a previous instance about the events in this task so far:\n"
                    f"<message>{summary_text}</message>"
                )

                # Reset messages but keep system prompt
                new_messages = []
                if state["messages"][0]["role"] == "system":
                    new_messages.append(state["messages"][0])
                new_messages.append({"role": "user", "content": compressed_user_msg})
                state["history_for_logging"].append(
                    {"role": "user", "content": compressed_user_msg}
                )

                # Update state for new segment
                state["messages"] = new_messages
                state["is_compressing"] = False
                state["has_been_compressed"] = True
                state["compression_count"] += 1
                state["current_segment_start"] = len(
                    state["completion_ids"]
                )  # Mark start of new segment

                # Reset prompt_ids for the new segment
                state["prompt_ids"] = []  # Will be set on next LLM call

                # Don't increment steps for compression
                return j, state

            # Parse action and execute gym step
            env_id = state["gym_env_id"]
            env_info = self._gym_envs[env_id]

            action = self._parse_action(assistant_msg["content"])

            if action is None:
                # Invalid action - terminate
                env_info["done"] = True
                state["completed"] = True
                state["episode_outcome"] = "invalid_action"
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
                        if reward > 0:
                            state["episode_outcome"] = "goal_reached"
                        else:
                            state["episode_outcome"] = "fell_in_hole"
                    else:
                        # Check if we need compression
                        current_segment_length = (
                            len(state["completion_ids"])
                            - state["current_segment_start"]
                        )
                        if (
                            self.use_context_compression
                            and current_segment_length
                            >= self.compression_threshold * self.max_completion_length
                            and state["compression_count"] < 3
                        ):  # Limit compressions to avoid infinite loops

                            # Add compression prompt
                            state["messages"].append(
                                {
                                    "role": "user",
                                    "content": self.compression_prompt_template,
                                }
                            )
                            state["history_for_logging"].append(
                                {
                                    "role": "user",
                                    "content": self.compression_prompt_template,
                                }
                            )
                            state["is_compressing"] = True
                        else:
                            # Continue episode - add next state
                            env_msg = {
                                "role": "user",
                                "content": self._state_to_description(
                                    next_state, env_info["grid"]
                                ),
                            }
                            state["messages"].append(env_msg)
                            state["history_for_logging"].append(env_msg)

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
            # TODO: should this be done????? or just let it be handled by the trainer?
            # if len(state["completion_ids"]) > sampling_params.max_tokens:
            #     state["completed"] = True
            #     # Truncate only the part after current segment start
            #     max_segment_tokens = (
            #         sampling_params.max_tokens - state["current_segment_start"]
            #     )
            #     segment_ids = state["completion_ids"][state["current_segment_start"] :][
            #         :max_segment_tokens
            #     ]
            #     segment_mask = state["completion_mask"][
            #         state["current_segment_start"] :
            #     ][:max_segment_tokens]

            #     # Update the full arrays
            #     state["completion_ids"] = (
            #         state["completion_ids"][: state["current_segment_start"]]
            #         + segment_ids
            #     )
            #     state["completion_mask"] = (
            #         state["completion_mask"][: state["current_segment_start"]]
            #         + segment_mask
            #     )

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

            if env_result.get("use_segments", False):
                # Handle segmented data from context compression
                prompt_ids_list = env_result["prompt_ids"]
                prompt_masks_list = env_result["prompt_masks"]
                completion_ids_list = env_result["completion_ids"]
                completion_masks_list = env_result["completion_masks"]
                completion_messages = env_result["messages"]
                history_for_logging = env_result["history_for_logging"]
                episode_outcomes = env_result.get(
                    "episode_outcomes", [None] * len(all_prompts)
                )
                compression_info = env_result.get(
                    "compression_info", [{}] * len(all_prompts)
                )
            else:
                # Convert to list format for compatibility
                prompt_ids_list = [
                    [prompt_ids[i].tolist()] for i in range(len(prompts))
                ]
                prompt_masks_list = [
                    [prompt_mask[i].tolist()] for i in range(len(prompts))
                ]
                completion_ids_list = [[ids] for ids in env_result["ids"]]
                completion_masks_list = [[mask] for mask in env_result["mask"]]
                completion_messages = env_result["messages"]
                history_for_logging = env_result["history_for_logging"]
                episode_outcomes = env_result.get(
                    "episode_outcomes", [None] * len(all_prompts)
                )
                compression_info = [{}] * len(all_prompts)
        else:
            prompt_ids_list = [None] * len(all_prompts)
            prompt_masks_list = [None] * len(all_prompts)
            completion_ids_list = [None] * len(all_prompts)
            completion_masks_list = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            history_for_logging = [None] * len(all_prompts)
            episode_outcomes = [None] * len(all_prompts)
            compression_info = [None] * len(all_prompts)

        # Broadcast all data
        prompt_ids_list = broadcast_object_list(prompt_ids_list, from_process=0)
        prompt_masks_list = broadcast_object_list(prompt_masks_list, from_process=0)
        completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)
        completion_masks_list = broadcast_object_list(
            completion_masks_list, from_process=0
        )
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        history_for_logging = broadcast_object_list(history_for_logging, from_process=0)
        episode_outcomes = broadcast_object_list(episode_outcomes, from_process=0)
        compression_info = broadcast_object_list(compression_info, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        # Slice for local process
        prompt_ids_list = prompt_ids_list[process_slice]
        prompt_masks_list = prompt_masks_list[process_slice]
        completion_ids_list = completion_ids_list[process_slice]
        completion_masks_list = completion_masks_list[process_slice]
        completion_messages = completion_messages[process_slice]
        history_for_logging = history_for_logging[process_slice]
        episode_outcomes = episode_outcomes[process_slice]
        compression_info = compression_info[process_slice]

        # For context compression, we need to handle old_per_token_logps differently
        if self.use_context_compression and any(
            len(segments) > 1 for segments in completion_ids_list
        ):
            # We'll compute old_per_token_logps in _compute_loss for each segment
            old_per_token_logps = None
            ref_per_token_logps = None
        else:
            # Original path for non-compression
            # Pad and concatenate for standard processing
            completion_ids = [
                torch.tensor(ids[0], device=device) for ids in completion_ids_list
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )

            completion_mask = [
                torch.tensor(mask[0], device=device) for mask in completion_masks_list
            ]
            completion_mask = pad(completion_mask, padding_value=0)

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            logits_to_keep = completion_ids.size(1)

            # Compute logps
            with torch.no_grad():
                if self.num_iterations > 1:
                    old_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
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
            # Add episode outcomes and compression info to reward kwargs
            reward_kwargs["episode_outcomes"] = episode_outcomes
            reward_kwargs["compression_info"] = compression_info
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

        # Calculate completion length across all segments
        total_completion_length = 0
        for masks_list in completion_masks_list:
            for mask in masks_list:
                total_completion_length += sum(mask)

        completion_length = (
            self.accelerator.gather_for_metrics(
                torch.tensor(total_completion_length, device=device)
            )
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
            history_for_logging_to_log = gather_object(history_for_logging)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        [str(prompts_to_log[0][-1]["content"])],
                        [history_for_logging_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):

                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "history_for_logging": history_for_logging_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        # Log compression-specific metrics
        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and self.use_context_compression
        ):
            # Log compression statistics
            compression_stats = []
            for i, comp_info in enumerate(compression_info):
                if comp_info.get("needed_compression", False):
                    compression_stats.append(
                        {
                            "episode": i,
                            "compression_count": comp_info.get("compression_count", 0),
                            "final_reward": (
                                rewards_to_log[i] if i < len(rewards_to_log) else 0.0
                            ),
                        }
                    )

            if compression_stats and self.accelerator.is_main_process:
                # Log average compression performance
                avg_reward_compressed = sum(
                    s["final_reward"] for s in compression_stats
                ) / len(compression_stats)
                avg_reward_uncompressed = sum(
                    rewards_to_log[i]
                    for i in range(len(compression_info))
                    if not compression_info[i].get("needed_compression", False)
                ) / max(1, len(compression_info) - len(compression_stats))

                print(f"\n[Step {self.state.global_step}] Compression Statistics:")
                print(
                    f"  Episodes requiring compression: {len(compression_stats)}/{len(compression_info)}"
                )
                print(f"  Avg reward (compressed): {avg_reward_compressed:.3f}")
                print(f"  Avg reward (uncompressed): {avg_reward_uncompressed:.3f}")

        return {
            "prompt_ids": (
                prompt_ids_list if self.use_context_compression else prompt_ids
            ),
            "prompt_mask": (
                prompt_masks_list if self.use_context_compression else prompt_mask
            ),
            "completion_ids": (
                completion_ids_list if self.use_context_compression else completion_ids
            ),
            "completion_mask": (
                completion_masks_list
                if self.use_context_compression
                else completion_mask
            ),
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "compression_info": compression_info,
        }

    def _compute_loss(self, model, inputs):
        """Override to handle context compression with multiple conversation segments."""

        # Check if we have segmented data (arrays of conversation segments)
        if (
            self.use_context_compression
            and isinstance(inputs["prompt_ids"], list)
            and any(isinstance(p, list) for p in inputs["prompt_ids"])
        ):
            if DEBUG:
                print(f"Inputs prompt_ids shape: {inputs['prompt_ids'].shape}")
                print(f"Inputs completion_ids shape: {inputs['completion_ids'].shape}")
                print(f"Inputs prompt_mask shape: {inputs['prompt_mask'].shape}")
                print(
                    f"Inputs completion_mask shape: {inputs['completion_mask'].shape}"
                )
                print(f"Inputs advantages shape: {inputs['advantages'].shape}")

            # Process each episode's segments
            total_loss = 0.0
            total_tokens = 0
            total_kl = 0.0
            total_kl_tokens = 0
            clip_counts = {"low": 0, "high": 0, "region": 0}
            total_clip_tokens = 0

            device = self.accelerator.device
            advantages = inputs["advantages"]

            for episode_idx in range(len(inputs["prompt_ids"])):
                episode_segments = len(inputs["prompt_ids"][episode_idx])
                episode_advantage = advantages[episode_idx]

                # Process each conversation segment
                for segment_idx in range(episode_segments):
                    # Extract segment data
                    segment_prompt_ids = torch.tensor(
                        inputs["prompt_ids"][episode_idx][segment_idx], device=device
                    ).unsqueeze(
                        0
                    )  # Add batch dimension
                    segment_prompt_mask = torch.tensor(
                        inputs["prompt_mask"][episode_idx][segment_idx], device=device
                    ).unsqueeze(0)
                    segment_completion_ids = torch.tensor(
                        inputs["completion_ids"][episode_idx][segment_idx],
                        device=device,
                    ).unsqueeze(0)
                    segment_completion_mask = torch.tensor(
                        inputs["completion_mask"][episode_idx][segment_idx],
                        device=device,
                    ).unsqueeze(0)

                    # Skip empty segments
                    if segment_completion_ids.size(1) == 0:
                        continue

                    # Concatenate prompt and completion
                    input_ids = torch.cat(
                        [segment_prompt_ids, segment_completion_ids], dim=1
                    )
                    attention_mask = torch.cat(
                        [segment_prompt_mask, segment_completion_mask], dim=1
                    )
                    logits_to_keep = segment_completion_ids.size(1)

                    # Get per-token log probabilities
                    per_token_logps = self._get_per_token_logps(
                        model, input_ids, attention_mask, logits_to_keep, batch_size=1
                    )

                    # Compute old log probs for this segment
                    with torch.no_grad():
                        if self.num_iterations > 1:
                            old_per_token_logps = self._get_per_token_logps(
                                self.model,
                                input_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=1,
                            )
                        else:
                            old_per_token_logps = per_token_logps.detach()

                    # Compute KL divergence if needed
                    per_token_kl = None
                    if self.beta != 0.0:
                        with torch.no_grad():
                            if self.ref_model is not None:
                                ref_per_token_logps = self._get_per_token_logps(
                                    self.ref_model,
                                    input_ids,
                                    attention_mask,
                                    logits_to_keep,
                                    batch_size=1,
                                )
                            else:
                                with self.accelerator.unwrap_model(
                                    self.model
                                ).disable_adapter():
                                    ref_per_token_logps = self._get_per_token_logps(
                                        self.model,
                                        input_ids,
                                        attention_mask,
                                        logits_to_keep,
                                        batch_size=1,
                                    )
                        per_token_kl = (
                            torch.exp(ref_per_token_logps - per_token_logps)
                            - (ref_per_token_logps - per_token_logps)
                            - 1
                        )

                    # Compute GRPO loss for this segment - using parent's epsilon values
                    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
                    coef_2 = torch.clamp(
                        coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high
                    )

                    per_token_loss1 = coef_1 * episode_advantage
                    per_token_loss2 = coef_2 * episode_advantage
                    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

                    if self.beta != 0.0 and per_token_kl is not None:
                        per_token_loss = per_token_loss + self.beta * per_token_kl

                    # Apply loss based on type
                    segment_mask = segment_completion_mask.squeeze(0)
                    if self.loss_type == "grpo":
                        segment_loss = (
                            per_token_loss * segment_mask
                        ).sum() / segment_mask.sum().clamp(min=1.0)
                    elif self.loss_type == "bnpo":
                        segment_loss = (
                            per_token_loss * segment_mask
                        ).sum() / segment_mask.sum().clamp(min=1.0)
                    elif self.loss_type == "dr_grpo":
                        segment_loss = (
                            per_token_loss * segment_mask
                        ).sum() / self.max_completion_length
                    else:
                        raise ValueError(f"Unknown loss type: {self.loss_type}")

                    # Accumulate metrics
                    total_loss += segment_loss * segment_mask.sum()
                    total_tokens += segment_mask.sum()

                    # Track KL divergence
                    if per_token_kl is not None:
                        total_kl += (per_token_kl * segment_mask).sum()
                        total_kl_tokens += segment_mask.sum()

                    # Track clipping statistics
                    is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (
                        episode_advantage < 0
                    )
                    is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
                        episode_advantage > 0
                    )
                    is_region_clipped = is_low_clipped | is_high_clipped

                    clip_counts["low"] += (is_low_clipped * segment_mask).sum().item()
                    clip_counts["high"] += (is_high_clipped * segment_mask).sum().item()
                    clip_counts["region"] += (
                        (is_region_clipped * segment_mask).sum().item()
                    )
                    total_clip_tokens += segment_mask.sum().item()

            # Average loss across all tokens
            loss = total_loss / total_tokens.clamp(min=1.0)

            # Log metrics
            mode = "train" if self.model.training else "eval"

            # Log KL divergence if computed
            if self.beta != 0.0 and total_kl_tokens > 0:
                mean_kl = total_kl / total_kl_tokens
                self._metrics[mode]["kl"].append(
                    self.accelerator.gather(mean_kl).nanmean().item()
                )

            # Log clipping statistics
            if total_clip_tokens > 0:
                low_clip_ratio = clip_counts["low"] / total_clip_tokens
                high_clip_ratio = clip_counts["high"] / total_clip_tokens
                region_clip_ratio = clip_counts["region"] / total_clip_tokens

                gathered_low_clip = self.accelerator.gather(
                    torch.tensor(low_clip_ratio, device=device)
                )
                self._metrics[mode]["clip_ratio/low_mean"].append(
                    gathered_low_clip.nanmean().item()
                )
                self._metrics[mode]["clip_ratio/low_min"].append(
                    nanmin(gathered_low_clip).item()
                )

                gathered_high_clip = self.accelerator.gather(
                    torch.tensor(high_clip_ratio, device=device)
                )
                self._metrics[mode]["clip_ratio/high_mean"].append(
                    gathered_high_clip.nanmean().item()
                )
                self._metrics[mode]["clip_ratio/high_max"].append(
                    nanmax(gathered_high_clip).item()
                )

                gathered_clip_ratio = self.accelerator.gather(
                    torch.tensor(region_clip_ratio, device=device)
                )
                self._metrics[mode]["clip_ratio/region_mean"].append(
                    gathered_clip_ratio.nanmean().item()
                )

            # Log compression-specific metrics
            compression_info = inputs.get("compression_info", [])
            if compression_info:
                num_compressed = sum(
                    1
                    for info in compression_info
                    if info.get("needed_compression", False)
                )
                compression_rate = num_compressed / len(compression_info)
                self._metrics[mode]["compression_rate"].append(compression_rate)

                avg_compressions = sum(
                    info.get("compression_count", 0) for info in compression_info
                ) / len(compression_info)
                self._metrics[mode]["avg_compressions_per_episode"].append(
                    avg_compressions
                )

            return loss
        else:
            # Fall back to parent implementation for non-compression cases
            return super()._compute_loss(model, inputs)

    def __del__(self):
        """Clean up gym environments when trainer is destroyed."""
        for env_info in self._gym_envs.values():
            if "env" in env_info:
                env_info["env"].close()

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Override to handle list-based inputs when using context compression."""

        # For context compression, we need to handle the generation differently
        # because the outputs are lists of segments, not tensors that can be split
        if self.use_context_compression:
            mode = "train" if self.model.training else "eval"

            # Check if we've already generated (outputs will have completion_ids)
            already_generated = "completion_ids" in inputs and isinstance(
                inputs.get("completion_ids"), list
            )

            if mode == "train":
                # During training, generate once per gradient_accumulation_steps * num_iterations
                # This matches the parent class behavior in TRL 0.17.1
                generate_every = (
                    self.args.gradient_accumulation_steps * self.num_iterations
                )

                if not already_generated and (
                    self._step % generate_every == 0 or self._buffered_inputs is None
                ):
                    # Generate completions
                    inputs = self._generate_and_score_completions(inputs)
                    self._buffered_inputs = inputs
                elif not already_generated:
                    # Use buffered inputs
                    inputs = self._buffered_inputs

                self._step += 1
            else:
                # In evaluation mode
                if not already_generated:
                    inputs = self._generate_and_score_completions(inputs)

            return inputs
        else:
            # Fall back to parent implementation for non-compression cases
            return super()._prepare_inputs(inputs)
