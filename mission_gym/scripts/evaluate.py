#!/usr/bin/env python3
"""Evaluate a trained model with visualization."""

import argparse
import sys
import time


def main():
    """Run evaluation with rendering."""
    parser = argparse.ArgumentParser(description="Evaluate trained Mission Gym agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (without .zip extension)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Render the environment (default: True)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mission Gym - Model Evaluation")
    print("=" * 60)
    
    # Import dependencies
    try:
        from stable_baselines3 import PPO
        from mission_gym.env import MissionGymEnv
    except ImportError as e:
        print(f"Import failed: {e}")
        return 1
    
    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        model = PPO.load(args.model)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return 1
    
    # Create environment
    render_mode = "human" if args.render else None
    env = MissionGymEnv(render_mode=render_mode)
    
    print(f"\nRunning {args.episodes} evaluation episodes...")
    print("-" * 40)
    
    episode_rewards = []
    episode_lengths = []
    wins = 0
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            if args.render:
                env.render()
                time.sleep(0.02)  # Small delay for visualization
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        if terminated and info.get("capture_progress", 0) >= 20:
            wins += 1
            result = "WIN ✓"
        else:
            result = "LOSS"
        
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, length={step_count}, {result}")
    
    env.close()
    
    print("-" * 40)
    print(f"\nSummary:")
    print(f"  Mean reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"  Mean length: {sum(episode_lengths) / len(episode_lengths):.0f}")
    print(f"  Win rate: {wins}/{args.episodes} ({100 * wins / args.episodes:.0f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
