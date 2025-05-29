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
        print(f"State ID: {custom_state.get('_gym_state_id', 'Not found')}")

        # Test that we can get the gym state (internal method)
        try:
            gym_state_info = env.get_gym_state(custom_state)
            print(f"Gym state initialized: {gym_state_info['gym_state']}")
            print(f"Game done: {gym_state_info['done']}")
        except Exception as e:
            print(f"Error accessing gym state: {e}")
        print()

        # Test environment response WITH state
        print("Testing environment response (with proper state):")
        messages = [
            {"role": "system", "content": env.system_prompt},
            {"role": "user", "content": "Initial state"},
        ]

        # Initialize state properly
        state = {
            "messages": messages,
            "prompt_messages": len(messages),
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": [],
        }
        # Add custom state fields
        custom_state = env.initialize_custom_state(messages)
        state.update(custom_state)

        # Test initial response (no assistant message yet)
        try:
            response = env.env_response(messages, state=state)
            print("Initial response:")
            print(
                response["content"][:100] + "..."
                if len(response["content"]) > 100
                else response["content"]
            )
            print()
        except Exception as e:
            print(f"Error getting initial response: {e}")
            print()

        # Test with assistant action
        messages.append({"role": "assistant", "content": "1"})  # Move right
        state["messages"] = messages

        try:
            response = env.env_response(messages, state=state)
            print("Response after move 1 (right):")
            print(
                response["content"][:100] + "..."
                if len(response["content"]) > 100
                else response["content"]
            )
            print()
        except Exception as e:
            print(f"Error after move: {e}")
            print()

        # Test invalid move
        messages.append(
            {"role": "user", "content": response["content"]}
        )  # Add the response
        messages.append({"role": "assistant", "content": "invalid"})  # Invalid move
        state["messages"] = messages

        try:
            response = env.env_response(messages, state=state)
            print("Response after invalid move:")
            print(response["content"])
            print()
        except Exception as e:
            print(f"Error after invalid move: {e}")
            print()

        # Test is_completed with state
        print("Testing is_completed:")
        print(f"Completed (with state): {env.is_completed(messages, state=state)}")
        print(f"Completed (without state, fallback): {env.is_completed(messages)}")
        print()

    print("\nNote: Full testing requires running through the trainer")
    print("which properly manages state through the generate() and step() methods.")
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_frozenlake_env()
