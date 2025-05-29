from typing import List, Dict, Any
import gymnasium as gym
import numpy as np
from datasets import Dataset

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
        **kwargs,
    ):

        # Create initial dataset if none provided
        if dataset is None:
            dataset = self.create_initial_dataset()

        super().__init__(
            dataset=dataset, system_prompt=system_prompt, few_shot=few_shot, **kwargs
        )

        # Initialize the FrozenLake environment
        self.gym_env = gym.make(
            "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False
        )

        # Keep track of game states for each conversation
        self.game_states: Dict[str, Any] = {}

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

    def reset_game(self, conversation_id: str) -> int:
        """Reset the game and return initial state."""
        state, _ = self.gym_env.reset()
        self.game_states[conversation_id] = {
            "state": state,
            "done": False,
            "last_reward": 0.0,
            "game_rewards": [],  # Track rewards for this game
        }
        return state

    def get_grid_layout(self) -> List[List[str]]:
        """Extract grid layout from the gymnasium environment."""
        # Get the grid description from the environment
        desc = self.gym_env.desc

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
            conversation_id = str(hash(str(completion)))
            if conversation_id in self.game_states:
                rewards = self.game_states[conversation_id]["game_rewards"]
                # Return the final reward (1.0 if reached goal, 0.0 otherwise)
                final_reward = max(rewards) if rewards else 0.0
                game_rewards.append(final_reward)
            else:
                game_rewards.append(0.0)
        return game_rewards

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return [self.format_reward_func, self.game_reward_func]

    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return [1.0, 1.0]  # Format reward weight  # Game reward weight

    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """Check if the game is completed."""
        # Generate a unique conversation ID based on the message history
        conversation_id = str(hash(str(messages)))

        if conversation_id in self.game_states:
            return self.game_states[conversation_id]["done"]

        # If we don't have game state, it means we haven't started yet
        return False

    def env_response(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, str]:
        """Generate environment response after processing the assistant's action."""
        # Generate conversation ID
        conversation_id = str(hash(str(messages)))

        # Get the last assistant message
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant_msg = msg
                break

        # Initialize game if not exists
        if conversation_id not in self.game_states:
            initial_state = self.reset_game(conversation_id)
            return {
                "role": "user",
                "content": self.get_state_description(initial_state)
                + "\n\nWhat is your move?",
            }

        game_state = self.game_states[conversation_id]

        # If game is already done, return final message
        if game_state["done"]:
            # TODO: this should not be needed if we end the game!
            return {"role": "user", "content": "Game completed!"}

        # Parse action from assistant message
        if last_assistant_msg is None:
            return {"role": "user", "content": "Please provide a valid move (0-3)."}

        action = self.parse_action(last_assistant_msg["content"])

        if action is None:
            # Invalid action format - end the game
            game_state["done"] = True
            game_state["game_rewards"].append(-0.1)  # Format penalty
            # TODO: this should not be needed if we end the game!
            return {
                "role": "user",
                "content": "Invalid move format. Game over! Please provide a valid digit (0-3) as your move.",
            }

        # Execute action in the environment
        try:
            next_state, reward, terminated, truncated, _ = self.gym_env.step(action)
            done = terminated or truncated

            # Update game state
            game_state["state"] = next_state
            game_state["done"] = done
            game_state["last_reward"] = reward
            game_state["game_rewards"].append(reward)

            if done:
                # TODO: this should not be needed if we end the game!
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
