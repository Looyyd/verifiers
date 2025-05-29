from typing import List, Dict, Any
import gymnasium as gym
import numpy as np
from datasets import Dataset
from copy import deepcopy
import threading

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv


class FrozenLakeEnv(MultiTurnEnv):
    def __init__(
        self,
        dataset: Dataset | None = None,
        system_prompt: str = """You are an agent playing frozen lake.
You will be given grids by the user, propose the best move.
Possible moves are:
0: UP
1: RIGHT
2: DOWN
3: LEFT

The first digit in your message will be considered as your move.""",
        few_shot: List[Dict[str, str]] = [],
        is_slippery: bool = False,
        map_name: str = "4x4",
        **kwargs,
    ):

        # Store gym configuration
        self.is_slippery = is_slippery
        self.map_name = map_name

        # Thread-local storage for accessing state in env_response
        self._thread_local = threading.local()

        # Create initial dataset if none provided
        if dataset is None:
            dataset = self.create_initial_dataset()

        super().__init__(
            dataset=dataset, system_prompt=system_prompt, few_shot=few_shot, **kwargs
        )

    def create_initial_dataset(self, n_samples: int = 100) -> Dataset:
        """Create a dataset with initial FrozenLake states."""
        # For FrozenLake, we always start at position 0 (top-left)
        initial_state = 0

        # Create dataset entries
        data = []
        for i in range(n_samples):
            data.append(
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": self.get_state_description(initial_state),
                        }
                    ],
                }
            )

        return Dataset.from_list(data)

    def initialize_custom_state(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Initialize FrozenLake-specific state fields."""
        # Create a new gym environment for this conversation
        gym_env = gym.make(
            "FrozenLake-v1",
            desc=None,
            map_name=self.map_name,
            is_slippery=self.is_slippery,
        )
        state, _ = gym_env.reset()

        return {
            "gym_env": gym_env,
            "gym_state": state,
            "game_done": False,
            "game_rewards": [],
            "last_reward": 0.0,
        }

    def get_grid_layout(self) -> List[List[str]]:
        """Extract grid layout from a fresh gymnasium environment."""
        # Create a temporary environment just to get the grid layout
        temp_env = gym.make(
            "FrozenLake-v1",
            desc=None,
            map_name=self.map_name,
            is_slippery=self.is_slippery,
        )
        desc = temp_env.desc

        # Convert bytes to strings if necessary
        if isinstance(desc[0][0], bytes):
            grid = [[cell.decode("utf-8") for cell in row] for row in desc]
        else:
            grid = [[str(cell) for cell in row] for row in desc]

        return grid

    def state_to_coordinates(self, state: int) -> tuple[int, int]:
        """Convert state number to (row, col) coordinates."""
        grid = self.get_grid_layout()
        ncol = len(grid[0])
        row = state // ncol
        col = state % ncol
        return row, col

    def get_state_description(self, state: int, agent_symbol: str = "A") -> str:
        """Convert state to grid description - works for any grid size."""
        # Get the base grid layout
        grid = self.get_grid_layout()

        # Convert state to coordinates
        row, col = self.state_to_coordinates(state)

        # Create display grid with agent position
        display_grid = [row[:] for row in grid]  # Deep copy
        display_grid[row][col] = agent_symbol

        # Format as string
        grid_str = "Current grid state:\n"
        for grid_row in display_grid:
            grid_str += " ".join(grid_row) + "\n"

        # Add legend (customize based on symbols found in grid)
        symbols = set()
        for row in grid:
            symbols.update(row)
        symbols.add(agent_symbol)

        legend_map = {
            "S": "Start",
            "F": "Frozen",
            "H": "Hole",
            "G": "Goal",
            agent_symbol: "Agent",
        }

        legend_parts = []
        for symbol in sorted(symbols):
            if symbol in legend_map:
                legend_parts.append(f"{symbol}={legend_map[symbol]}")
            else:
                legend_parts.append(f"{symbol}=Unknown")

        grid_str += f"\nLegend: {', '.join(legend_parts)}"
        return grid_str

    def parse_action(self, message: str) -> int | None:
        """Parse action from assistant message. Returns None if invalid."""
        if len(message) == 0:
            return None

        # Find the first digit in the message
        for char in message:
            if char.isdigit() and char in "0123":
                return int(char)
        return None

    def step(self, states, llm, sampling_params):
        """Override step to handle gym state updates properly."""
        # Store states in thread-local storage for access in env_response
        self._thread_local.states = states

        # Call parent step method
        states = super().step(states, llm, sampling_params)

        # Clean up thread-local storage
        if hasattr(self._thread_local, "states"):
            delattr(self._thread_local, "states")

        return states

    def format_reward_func(
        self, completions: List[List[Dict[str, str]]], **kwargs: Any
    ) -> List[float]:
        """
        Reward function that checks if the response format is correct.
        Returns -0.1 if format is incorrect, 0 if correct.
        """
        rewards = []
        for completion in completions:
            # Get the last assistant message
            last_assistant_msg = None
            for msg in reversed(completion):
                if msg["role"] == "assistant":
                    last_assistant_msg = msg
                    break

            if last_assistant_msg is None:
                rewards.append(-0.1)
                continue

            content = last_assistant_msg["content"]
            # Check if the first character is a digit 0-3
            if len(content) > 0 and content[0].isdigit() and content[0] in "0123":
                rewards.append(0.0)  # Correct format
            else:
                rewards.append(-0.1)  # Incorrect format

        return rewards

    def game_reward_func(
        self, completions: List[List[Dict[str, str]]], **kwargs: Any
    ) -> List[float]:
        """
        Reward function that returns the game rewards from FrozenLake.
        Returns 1.0 if goal reached, 0.0 for other valid moves.
        """
        game_rewards = []

        for completion in completions:
            # Check if goal was reached by looking at environment responses
            goal_reached = False
            for msg in completion:
                if (
                    msg["role"] == "user"
                    and "Congratulations! You reached the goal!" in msg["content"]
                ):
                    goal_reached = True
                    break

            game_rewards.append(1.0 if goal_reached else 0.0)

        return game_rewards

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return [self.format_reward_func, self.game_reward_func]

    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return [1.0, 1.0]  # Format reward weight  # Game reward weight

    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """Check if the game is completed by looking for state in thread-local storage."""
        # Try to get state from thread-local storage
        if hasattr(self._thread_local, "states"):
            # Find the state that corresponds to these messages
            for state in self._thread_local.states:
                if state["messages"] is messages:
                    return state.get("game_done", False)

        # Fallback: check messages for completion indicators
        if len(messages) < 2:
            return False

        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg
                break

        if last_user_msg:
            content = last_user_msg["content"]
            return (
                "Game over!" in content
                or "Congratulations! You reached the goal!" in content
            )

        return False

    def env_response(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, str]:
        """Generate environment response after processing the assistant's action."""
        # Try to get the current state from thread-local storage
        current_state = None
        if hasattr(self._thread_local, "states"):
            # Find the state that corresponds to these messages
            for state in self._thread_local.states:
                if state["messages"] is messages:
                    current_state = state
                    break

        # Get the last assistant message
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant_msg = msg
                break

        # Check if this is the first call (no assistant messages yet)
        if last_assistant_msg is None:
            # Initial state
            return {
                "role": "user",
                "content": self.get_state_description(0) + "\n\nWhat is your move?",
            }

        # If we don't have access to state, return error
        if current_state is None:
            return {
                "role": "user",
                "content": "Error: Unable to access game state. Please try again.",
            }

        # Parse action from last assistant message
        action = self.parse_action(last_assistant_msg["content"])

        if action is None:
            # Invalid action format - end the game
            current_state["game_done"] = True
            current_state["game_rewards"].append(-0.1)
            return {
                "role": "user",
                "content": "Invalid move format. Game over! Please provide a valid digit (0-3) as your move.",
            }

        # Execute action in the gym environment
        gym_env = current_state["gym_env"]
        try:
            next_state, reward, terminated, truncated, _ = gym_env.step(action)
            done = terminated or truncated

            # Update state
            current_state["gym_state"] = next_state
            current_state["game_done"] = done
            current_state["last_reward"] = reward
            current_state["game_rewards"].append(reward)

            # Generate response based on game state
            if done:
                if reward > 0:
                    return {
                        "role": "user",
                        "content": self.get_state_description(next_state)
                        + "\n\nCongratulations! You reached the goal!",
                    }
                else:
                    return {
                        "role": "user",
                        "content": self.get_state_description(next_state)
                        + "\n\nGame over! You fell into a hole.",
                    }
            else:
                return {
                    "role": "user",
                    "content": self.get_state_description(next_state),
                }

        except Exception as e:
            return {
                "role": "user",
                "content": f"Error executing move: {str(e)}. Please try again.",
            }
