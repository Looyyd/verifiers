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
from verifiers.utils.nan_utils import nanmin, nanmax, nanstd


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


class GRPOFrozenLakeTrainer(GRPOTrainer):
    """
    A GRPO trainer specifically for FrozenLake environment.
    Inherits directly from GRPOTrainer to allow custom modifications.

    This trainer extends GRPOTrainer with:
    - Multi-turn FrozenLake environment interaction
    - Custom reward functions for format compliance and game success
    - Context compression for long episodes
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
        max_episode_steps: int = 50,  # TODO: this could be handled by env, right now we count the steps which is not really needed?
        frozen_tile_probability: float = 0.8,
        # Context compression parameters
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
        self.compression_threshold = compression_threshold
        self.compression_prompt_template = compression_prompt_template

        # Validate compression parameters
        if not (0.0 < self.compression_threshold <= 1.0):
            raise ValueError(
                f"compression_threshold must be between 0 and 1, got {self.compression_threshold}"
            )
        # TODO: could refactor this, sometimes compression will be without a prompt, just reset the history!
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

# Action response format
You should think about your move first using <think></think> tags, then give your final answer.
Put your final answer in \\boxed{}, for example \\boxed{0} for LEFT, \\boxed{1} for DOWN, etc.

Example format:
<think>
I need to analyze the current state and find the best path to the goal while avoiding holes...
</think>

\\boxed{2}

# Summarization format
You might also be asked to summarize the conversation so far.
In that case you should use the <think> tags to organize your thoughts, then put the summary outside the <think> tags.

Example format:
<think>
I need to summarize the conversation so far...
</think>
summary here ...
"""

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
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
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
            map_desc = generate_random_map(
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

        for i, completion in enumerate(completions):
            outcome = episode_outcomes[i]

            if outcome == "invalid_action":
                rewards.append(-0.1)
                continue

            if outcome == "compression_too_long":
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

            # Generate with the stored map descriptions
            if map_descs and i < len(map_descs):
                gym_env = gym.make(
                    "FrozenLake-v1",
                    desc=map_descs[i],
                    is_slippery=self.is_slippery,
                )
            else:
                raise ValueError("No map description provided")

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
                # Track conversation segments for proper loss computation
                "conversation_segments": [],
                "current_segment_start": 0,  # Track where current segment starts in completion_ids
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

        # Add the final segment.
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
                segment_prompt_ids = state["prompt_ids"]

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
                    raise RuntimeError(f"WARNING: Missing data for segment creation")

            else:
                raise RuntimeError(
                    f"WARNING: Episode completed with no segments captured. "
                    f"completion_ids length: {len(state['completion_ids'])}, "
                    f"current_segment_start: {state['current_segment_start']}"
                )

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

        completion_messages = [s["messages"][s["prompt_messages"] :] for s in states]
        history_for_logging = [s["history_for_logging"] for s in states]
        episode_outcomes = [s["episode_outcome"] for s in states]


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
                segment_prompt_ids = state["prompt_ids"]

                state["conversation_segments"].append(
                    {
                        "prompt_ids": segment_prompt_ids,
                        "completion_ids": segment_completion_ids,
                        "completion_mask": segment_completion_mask,
                    }
                )

                # Get current game state
                env_info = self._gym_envs[state["gym_env_id"]]

                # If compression message is too long, stop the episode
                # If the compression is too long, first of all it's not what we want
                # also if it is really really long, the next prompt might be longer than max_length, which shuts down vllm
                # TODO: put as constant
                # TODO: add rewards to state to add negative reward here, instead of having to check outcome
                if len(summary_text) > 1000:
                    env_info["done"] = True
                    state["completed"] = True
                    state["episode_outcome"] = "compression_too_long"
                    return j, state

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
                        if DEBUG:
                            print(f"Current segment length: {current_segment_length}")
                            print(
                                f"Compression threshold * max_completion_length: {self.compression_threshold * self.max_completion_length}"
                            )
                        if (
                            current_segment_length
                            >= self.compression_threshold * self.max_completion_length
                        ):
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
                            # If don't need compression continue episode - add next state
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
        else:
            prompt_ids_list = [None] * len(all_prompts)
            prompt_masks_list = [None] * len(all_prompts)
            completion_ids_list = [None] * len(all_prompts)
            completion_masks_list = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            history_for_logging = [None] * len(all_prompts)
            episode_outcomes = [None] * len(all_prompts)

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

        # Count total segments across all episodes locally
        local_segment_count = sum(
            len(prompt_ids_list[ep]) for ep in range(len(prompt_ids_list))
        )

        # Find global max segments
        local_count_tensor = torch.tensor(local_segment_count, device=device)
        all_counts = self.accelerator.gather(local_count_tensor)
        max_segment_count = all_counts.max().item()

        # Now process segments with synchronized forward passes
        segment_idx_global = 0
        old_per_token_logps_list = []
        ref_per_token_logps_list = []

        for episode_idx in range(len(prompt_ids_list)):
            episode_old_logps = []
            episode_ref_logps = []

            for segment_idx in range(len(prompt_ids_list[episode_idx])):
                # Process real segment
                if segment_idx_global < local_segment_count:
                    prompt_ids_segment = torch.tensor(
                        prompt_ids_list[episode_idx][segment_idx], device=device
                    )
                    prompt_mask_segment = torch.tensor(
                        prompt_masks_list[episode_idx][segment_idx], device=device
                    )
                    completion_ids_segment = torch.tensor(
                        completion_ids_list[episode_idx][segment_idx], device=device
                    )
                    completion_mask_segment = torch.tensor(
                        completion_masks_list[episode_idx][segment_idx], device=device
                    )

                    # Skip empty completions
                    if len(completion_ids_list[episode_idx][segment_idx]) == 0:
                        episode_old_logps.append([])
                        episode_ref_logps.append([])
                        segment_idx_global += 1
                        continue

                    # Concatenate for this segment
                    input_ids_segment = torch.cat(
                        [
                            prompt_ids_segment.unsqueeze(0),
                            completion_ids_segment.unsqueeze(0),
                        ],
                        dim=1,
                    )
                    attention_mask_segment = torch.cat(
                        [
                            prompt_mask_segment.unsqueeze(0),
                            completion_mask_segment.unsqueeze(0),
                        ],
                        dim=1,
                    )
                    logits_to_keep = completion_ids_segment.size(0)

                    with torch.no_grad():
                        # Compute old log probs (if num_iterations > 1)
                        # TODO: could refactor this duplicate code with _compute_loss
                        if self.num_iterations > 1:
                            old_logps = self._get_per_token_logps(
                                self.model,
                                input_ids_segment,
                                attention_mask_segment,
                                logits_to_keep,
                            )
                            episode_old_logps.append(
                                old_logps.squeeze(0).cpu().tolist()
                            )
                        else:
                            # Will use per_token_logps.detach() in _compute_loss
                            episode_old_logps.append(None)

                        # Compute ref log probs (if beta > 0)
                        if self.beta != 0.0:
                            if self.ref_model is not None:
                                ref_logps = self._get_per_token_logps(
                                    self.ref_model,
                                    input_ids_segment,
                                    attention_mask_segment,
                                    logits_to_keep,
                                )
                            else:
                                with self.accelerator.unwrap_model(
                                    self.model
                                ).disable_adapter():
                                    ref_logps = self._get_per_token_logps(
                                        self.model,
                                        input_ids_segment,
                                        attention_mask_segment,
                                        logits_to_keep,
                                    )
                            episode_ref_logps.append(
                                ref_logps.squeeze(0).cpu().tolist()
                            )
                        else:
                            episode_ref_logps.append(None)
                segment_idx_global += 1

            old_per_token_logps_list.append(episode_old_logps)
            ref_per_token_logps_list.append(episode_ref_logps)

        # CRITICAL: Process dummy segments to match global max
        # this ensures same computation graph across all gpus
        while segment_idx_global < max_segment_count:
            # Create dummy tensors for forward pass, with 0 mask they will be ignored
            dummy_input = torch.zeros(1, 10, device=device, dtype=torch.long)
            dummy_mask = torch.zeros(1, 10, device=device, dtype=torch.long)

            with torch.no_grad():
                if self.num_iterations > 1:
                    _ = self._get_per_token_logps(
                        self.model, dummy_input, dummy_mask, 5
                    )
                if self.beta != 0.0:
                    if self.ref_model is not None:
                        _ = self._get_per_token_logps(
                            self.ref_model, dummy_input, dummy_mask, 5
                        )
                    else:
                        with self.accelerator.unwrap_model(
                            self.model
                        ).disable_adapter():
                            _ = self._get_per_token_logps(
                                self.model, dummy_input, dummy_mask, 5
                            )
            segment_idx_global += 1

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
            history_for_logging_to_log = gather_object(history_for_logging)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        [history_for_logging_to_log[0][0]["content"]],
                        [history_for_logging_to_log[0][1:]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )



        return {
            "prompt_ids": (prompt_ids_list),
            "prompt_mask": (prompt_masks_list),
            "completion_ids": (completion_ids_list),
            "completion_mask": (completion_masks_list),
            "old_per_token_logps": old_per_token_logps_list,
            "ref_per_token_logps": ref_per_token_logps_list,
            "advantages": advantages,
        }

    def _compute_loss(self, model, inputs):
        """Override to handle context compression with multiple conversation segments."""

        device = self.accelerator.device

        # Step 1: Flatten all segments locally
        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []
        all_advantages = []
        all_old_per_token_logps = []  # NEW: collect precomputed old logps
        all_ref_per_token_logps = []  # NEW: collect precomputed ref logps
        valid_segment_mask = []  # Track which segments are real vs padding

        # Check if we have precomputed logps
        has_old_logps = (
            inputs.get("old_per_token_logps") is not None
            and inputs["old_per_token_logps"]
        )
        has_ref_logps = (
            inputs.get("ref_per_token_logps") is not None
            and inputs["ref_per_token_logps"]
        )

        for episode_idx in range(len(inputs["prompt_ids"])):
            episode_advantage = inputs["advantages"][episode_idx]

            for segment_idx in range(len(inputs["prompt_ids"][episode_idx])):
                # Check if segment has completion tokens
                if len(inputs["completion_ids"][episode_idx][segment_idx]) > 0:
                    all_prompt_ids.append(
                        inputs["prompt_ids"][episode_idx][segment_idx]
                    )
                    all_prompt_masks.append(
                        inputs["prompt_mask"][episode_idx][segment_idx]
                    )
                    all_completion_ids.append(
                        inputs["completion_ids"][episode_idx][segment_idx]
                    )
                    all_completion_masks.append(
                        inputs["completion_mask"][episode_idx][segment_idx]
                    )
                    all_advantages.append(episode_advantage)
                    valid_segment_mask.append(True)

                    # NEW: Collect precomputed logps if available
                    if (
                        has_old_logps
                        and inputs["old_per_token_logps"][episode_idx][segment_idx]
                        is not None
                    ):
                        all_old_per_token_logps.append(
                            inputs["old_per_token_logps"][episode_idx][segment_idx]
                        )
                    if (
                        has_ref_logps
                        and inputs["ref_per_token_logps"][episode_idx][segment_idx]
                        is not None
                    ):
                        all_ref_per_token_logps.append(
                            inputs["ref_per_token_logps"][episode_idx][segment_idx]
                        )

        # Step 2: Find global max number of segments
        local_num_segments = len(all_prompt_ids)
        local_num_tensor = torch.tensor(local_num_segments, device=device)
        all_num_segments = self.accelerator.gather(local_num_tensor)
        max_num_segments = all_num_segments.max().item()

        # If no segments on any GPU, return zero loss
        if max_num_segments == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Step 3: Pad to global max segments
        while len(all_prompt_ids) < max_num_segments:
            # Add dummy segments
            all_prompt_ids.append([self.processing_class.pad_token_id])
            all_prompt_masks.append([0])
            all_completion_ids.append([self.processing_class.pad_token_id])
            all_completion_masks.append([0])
            all_advantages.append(0.0)
            valid_segment_mask.append(False)

            # NEW: Pad precomputed logps with zeros (matching the length of dummy completion)
            if has_old_logps and all_old_per_token_logps:
                # For dummy segments, add a single zero since completion has 1 token
                all_old_per_token_logps.append([0.0])
            if has_ref_logps and all_ref_per_token_logps:
                # For dummy segments, add a single zero since completion has 1 token
                all_ref_per_token_logps.append([0.0])

        # Step 4: Convert to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in all_prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.processing_class.pad_token_id)

        prompt_mask = [torch.tensor(mask, device=device) for mask in all_prompt_masks]
        prompt_mask = pad(prompt_mask, padding_value=0)

        completion_ids = [
            torch.tensor(ids, device=device) for ids in all_completion_ids
        ]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
        )

        completion_mask = [
            torch.tensor(mask, device=device) for mask in all_completion_masks
        ]
        completion_mask = pad(completion_mask, padding_value=0)

        advantages = torch.tensor(all_advantages, dtype=torch.float32, device=device)
        valid_segment_mask = torch.tensor(
            valid_segment_mask, dtype=torch.bool, device=device
        )

        # NEW: Process precomputed logps
        if has_old_logps and all_old_per_token_logps:
            old_per_token_logps = [
                torch.tensor(logps, device=device, dtype=torch.float32)
                for logps in all_old_per_token_logps
            ]
            old_per_token_logps = pad(old_per_token_logps, padding_value=0.0)
            # Ensure shape matches completion_ids
            if old_per_token_logps.shape[1] != completion_ids.shape[1]:
                raise ValueError(
                    f"Old per token logps shape {old_per_token_logps.shape[1]} does not match completion_ids shape {completion_ids.shape[1]}"
                )
        else:
            old_per_token_logps = None

        if has_ref_logps and all_ref_per_token_logps:
            ref_per_token_logps = [
                torch.tensor(logps, device=device, dtype=torch.float32)
                for logps in all_ref_per_token_logps
            ]
            ref_per_token_logps = pad(ref_per_token_logps, padding_value=0.0)
            # Ensure shape matches completion_ids
            if ref_per_token_logps.shape[1] != completion_ids.shape[1]:
                raise ValueError(
                    f"Ref per token logps shape {ref_per_token_logps.shape[1]} does not match completion_ids shape {completion_ids.shape[1]}"
                )
        else:
            ref_per_token_logps = None

        # Concatenate prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute per-token log probabilities
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Use precomputed old log probs if available, otherwise compute them
        if old_per_token_logps is None:
            with torch.no_grad():
                if self.num_iterations > 1:
                    old_per_token_logps = self._get_per_token_logps(
                        self.model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    old_per_token_logps = per_token_logps.detach()

        # Compute KL divergence if needed
        if self.beta != 0.0:
            # Use precomputed ref log probs if available, otherwise compute them
            if ref_per_token_logps is None:
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.ref_model, input_ids, attention_mask, logits_to_keep
                        )
                    else:
                        with self.accelerator.unwrap_model(
                            self.model
                        ).disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(
                                self.model, input_ids, attention_mask, logits_to_keep
                            )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute GRPO loss
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Apply valid segment mask to exclude padding from loss
        # Expand valid_segment_mask to match token dimension
        valid_mask_expanded = valid_segment_mask.unsqueeze(1).expand_as(completion_mask)
        masked_completion_mask = completion_mask * valid_mask_expanded

        # Compute final loss based on loss type
        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * masked_completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log metrics (only for valid segments)
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0 and masked_completion_mask.sum() > 0:
            mean_kl = (
                per_token_kl * masked_completion_mask
            ).sum() / masked_completion_mask.sum()

            gathered_kl = self.accelerator.gather_for_metrics(mean_kl)

            self._metrics[mode]["kl"].append(gathered_kl.nanmean().item())

        # Compute clipping metrics (only for valid segments)
        if masked_completion_mask.sum() > 0:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (
                advantages.unsqueeze(1) < 0
            )
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
                advantages.unsqueeze(1) > 0
            )
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = (
                is_low_clipped * masked_completion_mask
            ).sum() / masked_completion_mask.sum()
            high_clip = (
                is_high_clipped * masked_completion_mask
            ).sum() / masked_completion_mask.sum()
            clip_ratio = (
                is_region_clipped * masked_completion_mask
            ).sum() / masked_completion_mask.sum()

            gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(
                gathered_low_clip.nanmean().item()
            )
            self._metrics[mode]["clip_ratio/low_min"].append(
                nanmin(gathered_low_clip).item()
            )

            gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(
                gathered_high_clip.nanmean().item()
            )
            self._metrics[mode]["clip_ratio/high_max"].append(
                nanmax(gathered_high_clip).item()
            )

            gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(
                gathered_clip_ratio.nanmean().item()
            )

        return loss

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
        mode = "train" if self.model.training else "eval"

        # Check if we've already generated (outputs will have completion_ids)
        already_generated = "completion_ids" in inputs and isinstance(
            inputs.get("completion_ids"), list
        )

        if mode == "train":
            # During training, generate once per gradient_accumulation_steps * num_iterations
            # This matches the parent class behavior in TRL 0.17.1
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations

            if not already_generated and (
                self._step % generate_every == 0 or self._buffered_inputs is None
            ):
                # Generate completions
                generated_outputs = self._generate_and_score_completions(inputs)

                # For context compression, we need to manually split the list-based data
                # since split_tensor_dict expects tensors
                num_chunks = self.args.gradient_accumulation_steps
                batch_size = len(inputs) // num_chunks

                self._buffered_inputs = []
                for i in range(num_chunks):
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    chunk = {}

                    # Handle list-based fields (for segmented data)
                    for key in [
                        "prompt_ids",
                        "prompt_mask",
                        "completion_ids",
                        "completion_mask",
                    ]:
                        if key in generated_outputs and isinstance(
                            generated_outputs[key], list
                        ):
                            chunk[key] = generated_outputs[key][start_idx:end_idx]

                    # Handle tensor fields
                    for key in [
                        "advantages",
                        "old_per_token_logps",
                        "ref_per_token_logps",
                    ]:
                        if key in generated_outputs:
                            if isinstance(generated_outputs[key], torch.Tensor):
                                chunk[key] = generated_outputs[key][start_idx:end_idx]
                            else:
                                chunk[key] = generated_outputs[key]



                    self._buffered_inputs.append(chunk)

            elif not already_generated:
                # Use buffered inputs
                pass  # self._buffered_inputs is already set

            inputs = self._buffered_inputs[
                self._step % self.args.gradient_accumulation_steps
            ]
            self._step += 1
        else:
            # In evaluation mode
            if not already_generated:
                inputs = self._generate_and_score_completions(inputs)

        return inputs
