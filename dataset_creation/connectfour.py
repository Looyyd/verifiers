import numpy as np
import random
import json
from typing import List, Dict, Tuple, Optional
from datasets import Dataset
import os


class ConnectFourEnv:
    """Simple Connect Four environment for dataset generation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to empty board."""
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1  # 1 for player, -1 for opponent
        self.done = False
        self.winner = None
        return self.board.copy()

    def step(self, column: int) -> Tuple[np.ndarray, bool, Optional[int]]:
        """Make a move in the specified column."""
        if self.done:
            return self.board.copy(), True, self.winner

        # Check if column is valid
        if column < 0 or column > 6:
            return self.board.copy(), self.done, self.winner

        # Check if column is full
        if self.board[0, column] != 0:
            return self.board.copy(), self.done, self.winner

        # Find the lowest empty row in the column
        for row in range(5, -1, -1):
            if self.board[row, column] == 0:
                self.board[row, column] = self.current_player
                break

        # Check for win
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
        # Check for draw
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0

        # Switch player
        self.current_player = -self.current_player

        return self.board.copy(), self.done, self.winner

    def _check_win(self, player: int) -> bool:
        """Check if the player has won."""
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if all(self.board[row, col : col + 4] == player):
                    return True

        # Check vertical
        for col in range(7):
            for row in range(3):
                if all(self.board[row : row + 4, col] == player):
                    return True

        # Check diagonal (top-left to bottom-right)
        for row in range(3):
            for col in range(4):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return True

        # Check diagonal (bottom-left to top-right)
        for row in range(3, 6):
            for col in range(4):
                if all(self.board[row - i, col + i] == player for i in range(4)):
                    return True

        return False

    def get_valid_actions(self) -> List[int]:
        """Get list of valid columns to play in."""
        return [col for col in range(7) if self.board[0, col] == 0]

    def set_board(self, board: np.ndarray):
        """Set the board to a specific state."""
        self.board = board.copy()
        # Update done status
        self.done = False
        self.winner = None

        # Check if game is over
        if self._check_win(1):
            self.done = True
            self.winner = 1
        elif self._check_win(-1):
            self.done = True
            self.winner = -1
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0


def board_to_string(board: np.ndarray) -> str:
    """Convert board to string representation."""
    desc = "┌───┬───┬───┬───┬───┬───┬───┐\n"
    desc += "│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │\n"
    desc += "├───┼───┼───┼───┼───┼───┼───┤\n"

    for row in range(6):
        desc += "│"
        for col in range(7):
            if board[row, col] == 0:
                desc += " . │"
            elif board[row, col] == 1:
                desc += " X │"  # Player's pieces
            else:
                desc += " O │"  # Opponent's pieces
        desc += "\n"
        if row < 5:
            desc += "├───┼───┼───┼───┼───┼───┼───┤\n"

    desc += "└───┴───┴───┴───┴───┴───┴───┘"

    return desc


def generate_random_board_state(env: ConnectFourEnv, num_moves: int) -> np.ndarray:
    """Generate a random board state by playing random moves."""
    env.reset()

    for _ in range(num_moves):
        valid_actions = env.get_valid_actions()
        if not valid_actions or env.done:
            break

        action = random.choice(valid_actions)
        env.step(action)

    return env.board.copy()


def create_example(env: ConnectFourEnv) -> Dict[str, str]:
    """Create a single training example."""
    # Generate a random initial board state (0-20 moves played)
    num_initial_moves = random.randint(0, 20)
    initial_board = generate_random_board_state(env, num_initial_moves)

    # Set the environment to this state
    env.set_board(initial_board)

    # Determine current player based on number of pieces
    total_pieces = np.sum(np.abs(initial_board))
    # If even number of pieces, player 1 (X) goes next
    # If odd number of pieces, player -1 (O) goes next
    env.current_player = 1 if total_pieces % 2 == 0 else -1

    # Generate 1-3 random actions
    num_actions = random.randint(1, 3)
    actions = []
    action_descriptions = []

    for i in range(num_actions):
        valid_actions = env.get_valid_actions()
        if not valid_actions or env.done:
            break

        action = random.choice(valid_actions)
        actions.append(action)

        # Create action description
        player_symbol = "X" if env.current_player == 1 else "O"
        action_descriptions.append(
            f"{i+1}. Player {player_symbol} places in column {action}"
        )

        # Execute the action
        env.step(action)

    # If no actions were taken (game was already over), skip this example
    if not actions:
        return None

    # Create the final board state
    final_board = env.board.copy()

    # Create the prompt
    initial_board_str = board_to_string(initial_board)
    final_board_str = board_to_string(final_board)
    actions_str = "\n".join(action_descriptions)

    system_prompt = (
        "You will be given a Connect Four grid and a sequence of actions. "
        "Your task is to output the state of the grid after all actions have been taken.\n\n"
        "Grid representation:\n"
        "- .: Empty space\n"
        "- X: Player X's pieces\n"
        "- O: Player O's pieces\n"
        "- Columns are numbered 0-6 from left to right\n\n"
        "When a piece is placed in a column, it falls to the lowest available position in that column."
    )

    user_message = f"Current grid state:\n{initial_board_str}\n\nActions to take:\n{actions_str}\n\nWhat is the grid state after these actions?"

    assistant_message = (
        f"After applying the actions, the grid state is:\n{final_board_str}"
    )

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "completion": [
            {"role": "assistant", "content": assistant_message}
        ]
    }


def create_dataset(
    num_examples: int = 1000, save_path: str = "connectfour_grid_dataset"
):
    """Create the Connect Four grid visualization dataset."""
    print(
        f"Creating Connect Four grid visualization dataset with {num_examples} examples..."
    )

    env = ConnectFourEnv()
    dataset_examples = []

    # Generate examples
    attempts = 0
    while len(dataset_examples) < num_examples and attempts < num_examples * 2:
        attempts += 1

        example = create_example(env)
        if example is not None:
            dataset_examples.append(example)

        if len(dataset_examples) % 100 == 0 and len(dataset_examples) > 0:
            print(f"Generated {len(dataset_examples)}/{num_examples} examples...")

    print(f"Created {len(dataset_examples)} valid examples")

    # Create HuggingFace dataset
    dataset = Dataset.from_list(dataset_examples)

    # Save dataset
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)

    # Also save as JSONL for easy inspection
    jsonl_path = os.path.join(save_path, "dataset.jsonl")
    with open(jsonl_path, "w") as f:
        for example in dataset_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Dataset saved to {save_path}")
    print(f"JSONL version saved to {jsonl_path}")

    # Print a sample example
    print("\n" + "=" * 80)
    print("Sample example from the dataset:")
    print("=" * 80)
    sample = dataset_examples[0]
    
    print("\n[PROMPT]")
    for message in sample["prompt"]:
        print(f"\n[{message['role'].upper()}]")
        print(message["content"])
    
    print("\n[COMPLETION]")
    for message in sample["completion"]:
        print(f"\n[{message['role'].upper()}]")
        print(message["content"])

    return dataset


if __name__ == "__main__":
    # Create the dataset
    dataset = create_dataset(num_examples=1_000)

    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"Total examples: {len(dataset)}")

    # Analyze action distribution
    action_counts = {1: 0, 2: 0, 3: 0}
    for example in dataset:
        prompt = example["prompt"]
        user_msg = prompt[1]["content"]  # user message is second in prompt
        num_actions = user_msg.count("Player")
        if num_actions in action_counts:
            action_counts[num_actions] += 1

    print(f"Action distribution:")
    for num_actions, count in action_counts.items():
        print(
            f"  {num_actions} action(s): {count} examples ({count/len(dataset)*100:.1f}%)"
        )
