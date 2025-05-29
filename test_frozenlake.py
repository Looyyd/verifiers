#!/usr/bin/env python3
"""
Simple test script for the FrozenLake environment
"""

import verifiers as vf


def test_frozenlake_env():
    """Test the FrozenLake environment basic functionality."""
    print("Testing FrozenLake Environment")
    print("=" * 50)

    # Create environment
    env = vf.FrozenLakeEnv()

    # Test system prompt
    print("System prompt:")
    print(env.system_prompt)
    print()

    # Test dataset creation
    print(f"Dataset size: {len(env.dataset)}")
    print("Sample dataset entry:")
    print(env.dataset[0])
    print()

    # Test reward functions
    reward_funcs = env.get_reward_funcs()
    reward_weights = env.get_reward_weights()
    print(f"Number of reward functions: {len(reward_funcs)}")
    print(f"Reward weights: {reward_weights}")
    print()

    # Test format reward function
    print("Testing format reward function:")
    test_completions = [
        [{"role": "assistant", "content": "1"}],  # Valid format
        [{"role": "assistant", "content": "5"}],  # Invalid format (wrong digit)
        [{"role": "assistant", "content": "abc"}],  # Invalid format (not digit)
        [
            {"role": "assistant", "content": "2 I choose to go right"}
        ],  # Valid format (starts with digit)
    ]
    format_rewards = env.format_reward_func(test_completions)
    print(f"Format rewards: {format_rewards}")
    print()

    # Test environment response logic
    print("Testing environment response:")
    messages = [
        {"role": "system", "content": env.system_prompt},
        {"role": "user", "content": "Initial state"},
    ]

    # Test initial response (should initialize game)
    response = env.env_response(messages)
    print("Initial response:")
    print(response)
    print()

    # Test with assistant action
    messages.append({"role": "assistant", "content": "1"})  # Move right
    response = env.env_response(messages)
    print("Response after move 1 (right):")
    print(response)
    print()

    # Test invalid move
    messages.append({"role": "assistant", "content": "invalid"})  # Invalid move
    response = env.env_response(messages)
    print("Response after invalid move:")
    print(response)
    print()

    print("Test completed successfully!")


if __name__ == "__main__":
    test_frozenlake_env()
