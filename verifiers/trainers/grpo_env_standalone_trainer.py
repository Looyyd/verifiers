import warnings
from typing import Callable, Optional, Union, Any, List, Dict, Sequence
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random
import time
from abc import abstractmethod

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from peft import PeftConfig
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from verifiers import RewardFunc
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


class GRPOStandaloneMultiTurnTrainer(GRPOTrainer):
    """
    A standalone GRPO trainer with built-in multi-turn environment logic.
    This integrates the multiturn environment directly into the trainer for easier customization.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        scale_rewards: bool = False,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # Multi-turn specific parameters
        system_prompt: str = "",
        few_shot: List[Dict[str, str]] = [],
        max_workers: int = 10,
        max_steps: int = 10,
        sleep_time: float = 1.0,
        **kwargs,
    ):
        self.vllm_client = None
        if not args.use_vllm:  # type: ignore
            raise ValueError("vLLM must be enabled for GRPOStandaloneMultiTurnTrainer")
        if not (
            callable(reward_funcs)
            or (
                isinstance(reward_funcs, list)
                and all(callable(f) for f in reward_funcs)
            )
        ):
            raise ValueError(
                "reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models."
            )

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        # Multi-turn specific attributes
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.mask_env_response = mask_env_response
        self.max_workers = max_workers
        self.max_steps = max_steps
        self.sleep_time = sleep_time
        self.scale_rewards = scale_rewards

        # Token IDs - these should match your tokenizer
        self.eot_id = 151643
        self.message_end_id = 151645

        # Sampling parameters for generation
        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            repetition_penalty=self.repetition_penalty,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    @abstractmethod
    def is_completed(
        self,
        messages: List[Dict[str, str]],
        state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """
        Check if the conversation is completed.
        Override this in subclasses for specific completion logic.

        Args:
            messages: List of conversation messages
            state: Optional state dictionary with custom fields

        Returns:
            bool: True if conversation is completed
        """
        pass

    @abstractmethod
    def env_response(
        self,
        messages: List[Dict[str, str]],
        state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Generate environment response based on conversation history.
        Override this in subclasses for specific environment behavior.

        Args:
            messages: List of conversation messages
            state: Optional state dictionary with custom fields

        Returns:
            Dict with 'role' and 'content' for the environment response
        """
        pass

    def initialize_custom_state(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Override this method in subclasses to add custom state fields.
        Called when initializing a new state.
        """
        return {}

    def update_custom_state(
        self, state: Dict[str, Any], messages: List[Dict[str, str]]
    ) -> None:
        """
        Override this method in subclasses to update custom state fields.
        Called after each step.
        """
        pass

    def step(
        self,
        states: List[Dict[str, Any]],
        llm: LLM | VLLMClient,
        sampling_params: SamplingParams,
    ) -> List[Dict[str, Any]]:
        """Execute one step of multi-turn conversation for all active states."""

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
            state["messages"].append(
                {"role": "assistant", "content": llm_response.outputs[0].text}
            )

            # Get token lengths of env response and new completion
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # Update completion masks
            state["completion_mask"].extend([0] * env_response_len)
            state["completion_mask"].extend([1] * new_completion_len)

            # Update completion ids
            state["completion_ids"] = list(llm_response.prompt_token_ids)
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][
                len(state["prompt_ids"]) :
            ]

            # Handle message end tokens
            if (
                state["completion_ids"][-1] != 198
                and state["completion_ids"][-2] != self.message_end_id
            ):
                state["completion_ids"].append(self.message_end_id)
                state["completion_ids"].append(198)
                state["completion_mask"].append(1)
                state["completion_mask"].append(1)

            # Fix mask/id length mismatch
            if len(state["completion_ids"]) > len(state["completion_mask"]):
                state["completion_mask"].extend(
                    [1] * (len(state["completion_ids"]) - len(state["completion_mask"]))
                )
            if len(state["completion_mask"]) > len(state["completion_ids"]):
                state["completion_mask"] = state["completion_mask"][
                    : len(state["completion_ids"])
                ]

            # Check completion with state access
            if (
                self.is_completed(state["messages"], state=state)
                or len(state["completion_ids"]) > sampling_params.max_tokens - 1
            ):
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][
                    : sampling_params.max_tokens
                ]
                state["completion_mask"] = state["completion_mask"][
                    : len(state["completion_ids"])
                ]
            else:
                # Call env_response with state access
                state["messages"].append(
                    self.env_response(state["messages"], state=state)
                )
                # Update custom state after environment response
                self.update_custom_state(state, state["messages"])

            # Enforce that the completion mask and completion ids are the same length
            if not len(state["completion_mask"]) == len(state["completion_ids"]):
                print(f"Warning: mask/id length mismatch. Fixing...")
                min_len = min(
                    len(state["completion_mask"]), len(state["completion_ids"])
                )
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

    def generate_multiturn(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM | VLLMClient,
        sampling_params: SamplingParams,
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        """Generate multi-turn conversations with proper masking."""

        # Initialize state variables
        all_completed = False
        states = []
        for m in prompts:
            state = {
                "messages": m,
                "prompt_messages": len(m),
                "prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
            }
            # Add custom state fields from subclass
            custom_state = self.initialize_custom_state(m)
            state.update(custom_state)
            states.append(state)

        # Main loop
        step_count = 0
        while not all_completed and step_count < self.max_steps:
            states = self.step(states, llm, sampling_params)
            all_completed = all(state["completed"] for state in states)
            step_count += 1

        completion_messages = [s["messages"][s["prompt_messages"] :] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]

        return {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
        }

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate completions and score them with reward functions."""

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]  # type: ignore
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]  # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False  # type: ignore
        )  # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)  # type: ignore
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

        # Gather the original prompts in message dict form
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.generate_multiturn(
                prompts=all_prompts,
                llm=self.vllm_client,  # type: ignore
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result["ids"]
            completion_messages = env_result["messages"]
            completion_mask = env_result["mask"]
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)  # type: ignore

        completion_mask = [
            torch.tensor(mask, device=device) for mask in completion_mask
        ]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps
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

        # Use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]  # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}  # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)  # type: ignore

            output_reward_func = [
                reward if reward is not None else torch.nan
                for reward in output_reward_func
            ]
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        # If all reward functions return None for a given row, issue a warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}  # type: ignore
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]  # type: ignore
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)  # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)  # type: ignore
        advantages = rewards - mean_grouped_rewards

        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)  # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)  # type: ignore
        if self.scale_rewards:
            # Scale the rewards
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()  # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__  # type: ignore
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())  # type: ignore

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
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:  # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})  # type: ignore

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


# Example usage: A simple DoubleCheck trainer
class GRPODoubleCheckTrainer(GRPOStandaloneMultiTurnTrainer):
    """Example implementation for the DoubleCheck environment."""

    def is_completed(
        self,
        messages: List[Dict[str, str]],
        state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Check if conversation is completed - after 'Are you sure?' is asked."""
        return len(messages) > 1 and messages[-2]["content"] == "Are you sure?"

    def env_response(
        self,
        messages: List[Dict[str, str]],
        state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Always respond with 'Are you sure?'"""
        return {"role": "user", "content": "Are you sure?"}
