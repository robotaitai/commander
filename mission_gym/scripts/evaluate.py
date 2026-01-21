#!/usr/bin/env python3
"""Evaluate a trained model with visualization and detailed stats."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def find_model_path(model_arg: str) -> Path:
    """Find model path from argument (can be run name or direct path)."""
    # Try as direct path first
    direct = Path(model_arg)
    if direct.exists() or Path(f"{model_arg}.zip").exists():
        return direct
    
    # Try as run name
    from mission_gym.scripts.run_utils import get_runs_dir
    runs_dir = get_runs_dir()
    
    # Check for exact match
    run_dir = runs_dir / model_arg
    if run_dir.exists():
        model_path = run_dir / "final_model"
        if model_path.exists() or Path(f"{model_path}.zip").exists():
            return model_path
        # Check checkpoints
        checkpoints = list((run_dir / "checkpoints").glob("*.zip"))
        if checkpoints:
            return checkpoints[-1].with_suffix("")  # Latest checkpoint
    
    # Try partial match
    matching_runs = [d for d in runs_dir.iterdir() if d.is_dir() and model_arg in d.name]
    if len(matching_runs) == 1:
        run_dir = matching_runs[0]
        model_path = run_dir / "final_model"
        if model_path.exists() or Path(f"{model_path}.zip").exists():
            return model_path
    
    return direct  # Return original, let it fail with clear error


def main():
    """Run evaluation with rendering and detailed stats."""
    parser = argparse.ArgumentParser(description="Evaluate trained Mission Gym agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model or run name (e.g., 'swift-falcon-20260121' or 'runs/swift-falcon-20260121/final_model')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (headless mode)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Slow down rendering for better visibility",
    )
    args = parser.parse_args()
    
    # Import colored output utilities
    from mission_gym.scripts.run_utils import (
        Colors, print_banner, print_step, print_info, print_success,
        print_warning, print_error, print_divider,
    )
    c = Colors
    
    print()
    print(c.colorize("═" * 70, c.BRIGHT_CYAN))
    print(c.colorize("  🎮  ", c.BRIGHT_YELLOW) + c.colorize("MISSION GYM - MODEL EVALUATION", c.BOLD, c.BRIGHT_CYAN))
    print(c.colorize("═" * 70, c.BRIGHT_CYAN))
    print()
    
    # Find model path
    model_path = find_model_path(args.model)
    print(f"  {c.colorize('📁 Model:', c.BRIGHT_BLUE)} {model_path}")
    print()
    
    # Import dependencies
    try:
        from stable_baselines3 import PPO
        from mission_gym.env import MissionGymEnv
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return 1
    
    # Load model
    print_step(1, f"Loading model")
    try:
        model = PPO.load(str(model_path))
        print_info("Model loaded successfully")
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return 1
    
    # Create environment
    print()
    print_step(2, "Creating environment")
    render_mode = None if args.no_render else "human"
    env = MissionGymEnv(render_mode=render_mode)
    print_info(f"Render mode: {'disabled' if args.no_render else 'enabled'}")
    
    print()
    print_divider()
    print()
    print(f"  {c.colorize(f'🚀 Running {args.episodes} evaluation episodes...', c.BOLD, c.BRIGHT_GREEN)}")
    print()
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    episode_results = []
    wins = 0
    
    # Table header
    print(f"  {c.colorize('Ep', c.DIM):>5}  {c.colorize('Reward', c.DIM):>10}  {c.colorize('Steps', c.DIM):>8}  {c.colorize('Result', c.DIM):<12}  {c.colorize('Details', c.DIM)}")
    print(f"  {c.colorize('─' * 60, c.DIM)}")
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            if not args.no_render:
                env.render()
                if args.slow:
                    time.sleep(0.05)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        # Determine result
        capture_progress = info.get("capture_progress", 0)
        attackers_alive = info.get("attackers_alive", 0)
        defenders_alive = info.get("defenders_alive", 0)
        
        if terminated and capture_progress >= 20:
            wins += 1
            result = c.colorize("✓ WIN", c.BRIGHT_GREEN)
            result_str = "WIN"
        elif attackers_alive == 0:
            result = c.colorize("✗ LOST", c.BRIGHT_RED)
            result_str = "LOST"
        elif truncated:
            result = c.colorize("⏱ TIMEOUT", c.YELLOW)
            result_str = "TIMEOUT"
        else:
            result = c.colorize("? UNKNOWN", c.DIM)
            result_str = "UNKNOWN"
        
        # Color reward
        if total_reward > 50:
            reward_color = c.BRIGHT_GREEN
        elif total_reward > 0:
            reward_color = c.GREEN
        elif total_reward > -50:
            reward_color = c.YELLOW
        else:
            reward_color = c.RED
        
        details = f"cap:{capture_progress:.0f} atk:{attackers_alive} def:{defenders_alive}"
        
        print(f"  {ep+1:>5}  {c.colorize(f'{total_reward:>10.2f}', reward_color)}  {step_count:>8}  {result:<20}  {c.colorize(details, c.DIM)}")
        
        episode_results.append({
            "episode": ep + 1,
            "reward": total_reward,
            "steps": step_count,
            "result": result_str,
            "capture_progress": capture_progress,
            "attackers_alive": attackers_alive,
            "defenders_alive": defenders_alive,
        })
    
    env.close()
    
    # Summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    mean_length = np.mean(episode_lengths)
    win_rate = wins / args.episodes
    
    print()
    print_divider()
    print()
    print(f"  {c.colorize('📊 SUMMARY', c.BOLD, c.BRIGHT_CYAN)}")
    print()
    
    # Reward stats
    print(f"  {c.colorize('Reward:', c.BRIGHT_BLUE)}")
    print(f"     Mean:   {c.colorize(f'{mean_reward:>8.2f}', c.BRIGHT_GREEN if mean_reward > 0 else c.BRIGHT_RED)} ± {std_reward:.2f}")
    print(f"     Min:    {min_reward:>8.2f}")
    print(f"     Max:    {max_reward:>8.2f}")
    print()
    
    # Episode stats
    print(f"  {c.colorize('Episodes:', c.BRIGHT_BLUE)}")
    print(f"     Mean length:  {mean_length:.0f} steps")
    print(f"     Win rate:     {c.colorize(f'{win_rate*100:.0f}%', c.BRIGHT_GREEN if win_rate > 0.5 else c.YELLOW)} ({wins}/{args.episodes})")
    print()
    
    # Reward distribution bar
    print(f"  {c.colorize('Reward Distribution:', c.BRIGHT_BLUE)}")
    bins = [float('-inf'), -100, -50, 0, 50, 100, float('inf')]
    bin_labels = ['<-100', '-100~-50', '-50~0', '0~50', '50~100', '>100']
    bin_colors = [c.BRIGHT_RED, c.RED, c.YELLOW, c.GREEN, c.BRIGHT_GREEN, c.BRIGHT_CYAN]
    
    for i in range(len(bins) - 1):
        count = sum(1 for r in episode_rewards if bins[i] <= r < bins[i+1])
        bar_len = int(count / args.episodes * 30)
        bar = "█" * bar_len
        print(f"     {bin_labels[i]:>10}: {c.colorize(bar, bin_colors[i])} {count}")
    
    print()
    print_divider()
    print()
    
    # Save results
    if args.save_results:
        results = {
            "model": str(model_path),
            "episodes": args.episodes,
            "seed": args.seed,
            "deterministic": not args.stochastic,
            "summary": {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "min_reward": min_reward,
                "max_reward": max_reward,
                "mean_length": mean_length,
                "win_rate": win_rate,
                "wins": wins,
            },
            "episodes_detail": episode_results,
        }
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print_success(f"Results saved to {args.save_results}")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
