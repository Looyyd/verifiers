from abc import abstractmethod
from typing import List, Dict, Any, Tuple
import gymnasium as gym
from datasets import Dataset

from verifiers.envs.multiturn_env import MultiTurnEnv


class MultiTurnGymEnv(MultiTurnEnv):
    """Base class for multi-turn environments that use gymnasium."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        system_prompt: str = "",
        few_shot: List[Dict[str, str]] = [],
        **kwargs,
    ):
        super().__init__(
            dataset=dataset, system_prompt=system_prompt, few_shot=few_shot, **kwargs
        )

        # Store gym environments and states indexed by state ID
        self._gym_states: Dict[int, Dict[str, Any]] = {}
        self._next_state_id = 0

    @abstractmethod
    def make_gym_env(self) -> gym.Env:
        """Create a new gymnasium environment instance."""
        pass

    @abstractmethod
    def get_state_description(self, gym_state: Any) -> str:
        """Convert gymnasium state to text description."""
        pass

    @abstractmethod
    def parse_action(self, message: str) -> Any:
        """Parse action from assistant message."""
        pass

    @abstractmethod
    def process_gym_step(
        self, gym_env: gym.Env, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute action in gym environment and return step results."""
        pass

    def initialize_custom_state(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Initialize state with gym environment."""
        # Create unique state ID
        state_id = self._next_state_id
        self._next_state_id += 1

        # Create new gym environment
        gym_env = self.make_gym_env()
        initial_state, _ = gym_env.reset()

        # Store gym state
        self._gym_states[state_id] = {
            "gym_env": gym_env,
            "gym_state": initial_state,
            "done": False,
            "rewards": [],
            "last_reward": 0.0,
        }

        return {
            "_gym_state_id": state_id,
        }

    def get_gym_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get gym state from state dict."""
        state_id = state.get("_gym_state_id")
        if state_id is None or state_id not in self._gym_states:
            raise RuntimeError(
                f"Gym state not found for state_id={state_id}. "
                "This likely means the state management is broken."
            )
        return self._gym_states[state_id]

    def is_completed(
        self,
        messages: List[Dict[str, str]],
        state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Check if the game is completed."""
        # If we have access to state (from our custom step method), use it
        if state is not None:
            gym_state = self.get_gym_state(state)
            return gym_state["done"]

        # Fallback: check messages for completion indicators
        # This is used by parent's step method
        if len(messages) < 2:
            return False

        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg
                break

        if last_user_msg:
            # Subclasses should override this with their specific completion messages
            return self._is_completion_message(last_user_msg["content"])

        return False

    def _is_completion_message(self, content: str) -> bool:
        """Check if a message indicates completion. Override in subclasses."""
        return False

    def env_response(
        self,
        messages: List[Dict[str, str]],
        state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Generate environment response. Must have access to state."""
        if state is None:
            raise RuntimeError(
                "env_response called without state. This is a bug in the multi-turn gym environment."
            )

        gym_state_info = self.get_gym_state(state)

        # Get the last assistant message
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant_msg = msg
                break

        # Check if this is the first call (no assistant messages yet)
        if last_assistant_msg is None:
            # Return initial state
            return {
                "role": "user",
                "content": self.get_state_description(gym_state_info["gym_state"]),
            }

        # Parse action from assistant message
        action = self.parse_action(last_assistant_msg["content"])

        if action is None:
            # Invalid action - end the game
            gym_state_info["done"] = True
            return self._invalid_action_response()

        # Execute action in gym environment
        gym_env = gym_state_info["gym_env"]
        try:
            next_state, reward, terminated, truncated, info = self.process_gym_step(
                gym_env, action
            )
            done = terminated or truncated

            # Update state
            gym_state_info["gym_state"] = next_state
            gym_state_info["done"] = done
            gym_state_info["last_reward"] = reward
            gym_state_info["rewards"].append(reward)

            # Generate response based on game state
            return self._generate_step_response(next_state, reward, done, info)

        except Exception as e:
            raise RuntimeError(f"Error executing action in gym environment: {str(e)}")

    @abstractmethod
    def _invalid_action_response(self) -> Dict[str, str]:
        """Response for invalid actions."""
        pass

    @abstractmethod
    def _generate_step_response(
        self, state: Any, reward: float, done: bool, info: Dict
    ) -> Dict[str, str]:
        """Generate response after a step in the environment."""
        pass

    def step(self, states, llm, sampling_params):
        """Override step to properly pass state to is_completed and env_response."""
        from copy import deepcopy
        from concurrent.futures import ThreadPoolExecutor
        import random
        import time
        from verifiers.inference.vllm_client import VLLMClient
        from ..imports import LLM

        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]

        # Get LLM responses (same as parent)
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
            from verifiers.envs.multiturn_env import dict_to_chat_response

            llm_responses = dict_to_chat_response(llm_responses).responses
        else:
            llm_responses = llm.chat(
                messages_to_step, sampling_params=sampling_params, use_tqdm=False
            )

        def update_state(j, llm_response):
            # sleep for rate limiting
            time.sleep(self.sleep_time * random.random())

            state = deepcopy(states[j])
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids
            state["messages"].append(
                {"role": "assistant", "content": llm_response.outputs[0].text}
            )

            # Token length calculations (same as parent)
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # Update completion masks and ids (same as parent)
            state["completion_mask"].extend([self.env_mask] * env_response_len)
            state["completion_mask"].extend([1] * new_completion_len)

            state["completion_ids"] = list(llm_response.prompt_token_ids)
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][
                len(state["prompt_ids"]) :
            ]

            # Handle message end tokens (same as parent)
            if (
                state["completion_ids"][-1] != 198
                and state["completion_ids"][-2] != self.message_end_id
            ):
                state["completion_ids"].append(self.message_end_id)
                state["completion_ids"].append(198)
                state["completion_mask"].append(1)
                state["completion_mask"].append(1)

            # Fix mask/id length mismatch (same as parent)
            if len(state["completion_ids"]) > len(state["completion_mask"]):
                state["completion_mask"].extend(
                    [1] * (len(state["completion_ids"]) - len(state["completion_mask"]))
                )
            if len(state["completion_mask"]) > len(state["completion_ids"]):
                state["completion_mask"] = state["completion_mask"][
                    : len(state["completion_ids"])
                ]

            # Check completion WITH state access
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
                # Call env_response WITH state access
                state["messages"].append(
                    self.env_response(state["messages"], state=state)
                )
                # Update custom state if needed
                self.update_custom_state(state, state["messages"])

            # Handle tokenizer bug (same as parent)
            if not len(state["completion_mask"]) == len(state["completion_ids"]):
                print(state["messages"])
                print(state["completion_mask"])
                print(state["completion_ids"])
                min_len = min(
                    len(state["completion_mask"]), len(state["completion_ids"])
                )
                state["completion_mask"] = state["completion_mask"][:min_len]
                state["completion_ids"] = state["completion_ids"][:min_len]

            return j, state

        # Execute updates in parallel (same as parent)
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
