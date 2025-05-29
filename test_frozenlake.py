#!/usr/bin/env python3
"""
Simple test script for the FrozenLake environment
"""

import verifiers as vf


def test_frozenlake_env():
    """Test the FrozenLake environment basic functionality."""
    print("Testing FrozenLake Environment")
    print("=" * 50)

    # Test both deterministic and slippery environments
    for is_slippery in [False, True]:
        print(f"\nTesting with is_slippery={is_slippery}")
        print("-" * 30)

        # Create environment
        env = vf.FrozenLakeEnv(is_slippery=is_slippery)

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
                {"role": "assistant", "content": "2 I choose to go down"}
            ],  # Valid format (starts with digit)
        ]
        format_rewards = env.format_reward_func(test_completions)
        print(f"Format rewards: {format_rewards}")
        print()

        # Test custom state initialization
        print("Testing custom state initialization:")
        test_messages = [
            {"role": "system", "content": env.system_prompt},
            {"role": "user", "content": "Initial state"},
        ]
        custom_state = env.initialize_custom_state(test_messages)
        print(f"Custom state keys: {list(custom_state.keys())}")
        print(f"Initial gym state: {custom_state['gym_state']}")
        print(f"Game done: {custom_state['game_done']}")
        print()

        # Test environment response logic
        print("Testing environment response (without thread-local state):")
        messages = [
            {"role": "system", "content": env.system_prompt},
            {"role": "user", "content": "Initial state"},
        ]

        # Test initial response (should work without state)
        response = env.env_response(messages)
        print("Initial response:")
        print(
            response["content"][:100] + "..."
            if len(response["content"]) > 100
            else response["content"]
        )
        print()

        # Test with assistant action (will fail without thread-local state)
        messages.append({"role": "assistant", "content": "1"})  # Move right
        response = env.env_response(messages)
        print("Response after move 1 (without thread-local state):")
        print(response["content"])
        print()

    print("\nNote: Full testing requires running through the trainer")
    print("which properly manages state through the step() method.")
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_frozenlake_env()
