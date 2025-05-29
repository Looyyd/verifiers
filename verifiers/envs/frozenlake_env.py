from typing import List, Dict, Any, Tuple
import gymnasium as gym
from datasets import Dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_gym_env import MultiTurnGymEnv


class FrozenLakeEnv(MultiTurnGymEnv):
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

        # Create dataset entries that match the expected format
        data = []
        for i in range(n_samples):
            data.append(
                {
                    "question": self.get_initial_state_description(),
                    "answer": "",  # No specific answer for FrozenLake
                    "task": "frozenlake",
                }
            )

        return Dataset.from_list(data)

    def get_initial_state_description(self) -> str:
        """Get description of initial state without creating a gym env."""
        # For 4x4 FrozenLake, agent starts at position 0 (top-left)
        return self.get_state_description(0)

    def make_gym_env(self) -> gym.Env:
        """Create a new FrozenLake gymnasium environment."""
        return gym.make(
            "FrozenLake-v1",
            desc=None,
            map_name=self.map_name,
            is_slippery=self.is_slippery,
        )

    def get_grid_layout(self) -> List[List[str]]:
        """Extract grid layout from a fresh gymnasium environment."""
        temp_env = self.make_gym_env()
        # Access the underlying environment through the wrapper
        desc = temp_env.unwrapped.desc

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

    def get_state_description(self, gym_state: Any) -> str:
        """Convert gymnasium state to text description."""
        # gym_state is an integer representing position in FrozenLake
        state = int(gym_state)

        # Get the base grid layout
        grid = self.get_grid_layout()

        # Convert state to coordinates
        row, col = self.state_to_coordinates(state)

        # Create display grid with agent position
        display_grid = [row[:] for row in grid]  # Deep copy
        display_grid[row][col] = "A"  # Agent symbol

        # Format as string
        grid_str = "Current grid state:\n"
        for grid_row in display_grid:
            grid_str += " ".join(grid_row) + "\n"

        # Add legend
        grid_str += "\nLegend: A=Agent, S=Start, F=Frozen, H=Hole, G=Goal"
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

    def process_gym_step(
        self, gym_env: gym.Env, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute action in gym environment and return step results."""
        return gym_env.step(action)

    def _is_completion_message(self, content: str) -> bool:
        """Check if a message indicates completion."""
        return (
            "Game over!" in content
            or "Congratulations! You reached the goal!" in content
        )

    def _invalid_action_response(self) -> Dict[str, str]:
        """Response for invalid actions."""
        return {
            "role": "user",
            "content": "Invalid move format. Game over! Please provide a valid digit (0-3) as your move.",
        }

    def _generate_step_response(
        self, state: Any, reward: float, done: bool, info: Dict
    ) -> Dict[str, str]:
        """Generate response after a step in the environment."""
        if done:
            if reward > 0:
                return {
                    "role": "user",
                    "content": self.get_state_description(state)
                    + "\n\nCongratulations! You reached the goal!",
                }
            else:
                return {
                    "role": "user",
                    "content": self.get_state_description(state)
                    + "\n\nGame over! You fell into a hole.",
                }
        else:
            return {
                "role": "user",
                "content": self.get_state_description(state),
            }

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
