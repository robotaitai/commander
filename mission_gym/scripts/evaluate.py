#!/usr/bin/env python3
"""Evaluate a trained model with visualization and detailed stats."""

# Suppress TensorFlow/TensorBoard noise (MUST be first, before ANY imports)
import os
# Set environment variables before any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no info, 2=no warnings, 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'

import sys
import warnings
from io import StringIO

# Suppress all Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ImportWarning)
warnings.filterwarnings('ignore', message='.*tensorboard.*')
warnings.filterwarnings('ignore', message='.*matplotlib.*')

# Context manager to suppress stderr during noisy imports
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stderr = self._original_stderr
        return False

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np


def find_model_path(model_arg: str) -> Path:
    """Find model path from argument (can be run name or direct path)."""
    direct = Path(model_arg)
    if direct.exists() or Path(f"{model_arg}.zip").exists():
        return direct
    
    from mission_gym.scripts.run_utils import get_runs_dir
    runs_dir = get_runs_dir()
    
    run_dir = runs_dir / model_arg
    if run_dir.exists():
        model_path = run_dir / "final_model"
        if model_path.exists() or Path(f"{model_path}.zip").exists():
            return model_path
        checkpoints = list((run_dir / "checkpoints").glob("*.zip"))
        if checkpoints:
            return checkpoints[-1].with_suffix("")
    
    matching_runs = [d for d in runs_dir.iterdir() if d.is_dir() and model_arg in d.name]
    if len(matching_runs) == 1:
        run_dir = matching_runs[0]
        model_path = run_dir / "final_model"
        if model_path.exists() or Path(f"{model_path}.zip").exists():
            return model_path
    
    return direct


# Action name abbreviations for display
ACTION_ABBREV = {
    "NOOP": "---",
    "THROTTLE_UP": "THR+",
    "THROTTLE_DOWN": "THR-",
    "TURN_LEFT": "←TRN",
    "TURN_RIGHT": "TRN→",
    "YAW_LEFT": "←YAW",
    "YAW_RIGHT": "YAW→",
    "BRAKE": "BRK!",
    "HOLD": "HOLD",
    "ALT_UP": "ALT+",
    "ALT_DOWN": "ALT-",
    "TAG": "TAG!",
    "SCAN": "SCAN",
}


def get_action_color(action: str, c) -> str:
    """Get color for action display."""
    if action in ("THROTTLE_UP", "THR+"):
        return c.BRIGHT_GREEN
    elif action in ("THROTTLE_DOWN", "THR-", "BRAKE", "BRK!"):
        return c.YELLOW
    elif action in ("TURN_LEFT", "TURN_RIGHT", "YAW_LEFT", "YAW_RIGHT", "←TRN", "TRN→", "←YAW", "YAW→"):
        return c.CYAN
    elif action in ("TAG", "TAG!"):
        return c.BRIGHT_RED
    elif action in ("SCAN",):
        return c.MAGENTA
    elif action in ("ALT_UP", "ALT_DOWN", "ALT+", "ALT-"):
        return c.BRIGHT_BLUE
    elif action in ("HOLD",):
        return c.DIM
    else:
        return c.DIM


def main():
    """Run evaluation with rendering and detailed stats."""
    parser = argparse.ArgumentParser(description="Evaluate trained Mission Gym agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model or run name",
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
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show live commands being sent to each unit",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of past commands to show per unit (default: 5)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to config directory (default: infer from model path)",
    )
    args = parser.parse_args()
    
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
    
    model_path = find_model_path(args.model)
    print(f"  {c.colorize('📁 Model:', c.BRIGHT_BLUE)} {model_path}")
    
    # Infer or use provided config directory
    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        run_dir = model_path.parent
        config_dir = run_dir / "configs"
    
    if config_dir.exists():
        print(f"  {c.colorize('⚙️  Config:', c.BRIGHT_BLUE)} {config_dir}")
    else:
        print(f"  {c.colorize('⚠️  Config:', c.YELLOW)} {config_dir} (not found, using defaults)")
        config_dir = None
    print()
    
    try:
        # Suppress TensorBoard/TensorFlow errors during import
        with SuppressStderr():
            from stable_baselines3 import PPO
        from mission_gym.env import MissionGymEnv
        from mission_gym.config import FullConfig
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return 1
    
    print_step(1, f"Loading configuration")
    if config_dir:
        try:
            config = FullConfig.load(config_dir=config_dir)
            print_info("Configuration loaded from run directory")
        except Exception as e:
            print_warning(f"Failed to load config from {config_dir}: {e}")
            print_info("Using default configuration")
            config = None
    else:
        config = None
    
    # Print environment signature
    if config:
        num_attackers = len(config.scenario.attackers)
        num_defenders = len(config.scenario.defenders)
        capture_time = config.scenario.objective.capture_time_required
        obj_radius = config.scenario.objective.radius
        stagnation_sec = config.termination.stagnation_seconds
        tag_enabled = config.engagement.tag_enabled
        
        print()
        print(f"  {c.colorize('🎯 Env Signature:', c.BRIGHT_YELLOW)} "
              f"attackers={c.colorize(str(num_attackers), c.BRIGHT_GREEN)} "
              f"defenders={c.colorize(str(num_defenders), c.BRIGHT_CYAN)} "
              f"capture_time={c.colorize(f'{capture_time:.0f}s', c.YELLOW)} "
              f"obj_radius={c.colorize(f'{obj_radius:.0f}m', c.MAGENTA)} "
              f"stagnation={c.colorize(f'{stagnation_sec:.0f}s', c.RED)} "
              f"tag={c.colorize('ON' if tag_enabled else 'OFF', c.BRIGHT_GREEN if tag_enabled else c.DIM)}")
    
    print()
    print_step(2, f"Loading model")
    try:
        model = PPO.load(str(model_path))
        print_info("Model loaded successfully")
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return 1
    
    print()
    print_step(3, "Creating environment")
    render_mode = None if args.no_render else "human"
    env = MissionGymEnv(render_mode=render_mode, config_dir=config_dir)
    print_info(f"Render mode: {'disabled' if args.no_render else 'enabled'}")
    print_info(f"Verbose mode: {'ON - showing live commands' if args.verbose else 'OFF'}")
    
    print()
    print_divider()
    print()
    print(f"  {c.colorize(f'🚀 Running {args.episodes} evaluation episodes...', c.BOLD, c.BRIGHT_GREEN)}")
    if not args.no_render:
        print(f"  {c.colorize('Controls:', c.BRIGHT_BLUE)} Press 'R' to restart current episode, 'ESC' to quit")
    print()
    
    episode_rewards = []
    episode_lengths = []
    episode_results = []
    wins = 0
    
    # Get action names for each unit type
    def get_action_names(env):
        action_names = []
        for attacker in env.attackers:
            if attacker.type_config:
                action_names.append(attacker.type_config.actions)
            else:
                action_names.append(["NOOP"])
        return action_names
    
    ep = 0
    while ep < args.episodes:
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        step_count = 0
        
        # Get action names and unit info
        action_names = get_action_names(env)
        unit_names = [f"{a.unit_type}" for a in env.attackers]
        
        # Command history per unit
        command_history = [deque(maxlen=args.history) for _ in range(env.num_attackers)]
        
        # Action statistics
        action_counts = [{} for _ in range(env.num_attackers)]
        
        if args.verbose:
            print()
            print(f"  {c.colorize(f'═══ Episode {ep+1} ═══', c.BOLD, c.BRIGHT_YELLOW)}")
            print()
            # Print unit header
            header = "  Step  │ "
            for i, name in enumerate(unit_names):
                header += f" A{i}:{name[:6]:>6} │"
            header += " Reward"
            print(c.colorize(header, c.DIM))
            print(c.colorize("  " + "─" * (len(header) - 2), c.DIM))
        
        restart_requested = False
        while not done:
            # Check for keyboard events (restart, quit)
            if not args.no_render:
                try:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print()
                            print_warning("Window closed by user")
                            env.close()
                            return 0
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                print()
                                print_warning("ESC pressed - exiting evaluation")
                                env.close()
                                return 0
                            elif event.key == pygame.K_r:
                                print()
                                print_info("R pressed - restarting episode")
                                restart_requested = True
                                break
                except:
                    pass  # pygame not available or not initialized
            
            if restart_requested:
                break
            
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            
            # Decode actions to names
            current_commands = []
            for i, act_idx in enumerate(action):
                if i < len(action_names) and act_idx < len(action_names[i]):
                    cmd = action_names[i][act_idx]
                else:
                    cmd = "NOOP"
                current_commands.append(cmd)
                command_history[i].append(cmd)
                
                # Count actions
                action_counts[i][cmd] = action_counts[i].get(cmd, 0) + 1
            
            # Print live commands
            if args.verbose and step_count % 10 == 0:  # Print every 10 steps
                line = f"  {step_count:>4}  │ "
                for i, cmd in enumerate(current_commands):
                    abbrev = ACTION_ABBREV.get(cmd, cmd[:4])
                    color = get_action_color(cmd, c)
                    line += f" {c.colorize(f'{abbrev:>10}', color)} │"
                line += f" {total_reward:>6.1f}"
                print(line)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            if not args.no_render:
                env.render()
                if args.slow:
                    time.sleep(0.05)
        
        # If restart was requested, don't count this episode
        if restart_requested:
            continue  # Redo this episode (don't increment ep)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        # Get info with proper fallbacks (training uses namespaced keys)
        capture_progress = info.get("capture_progress", info.get("kpi/final_capture_progress", 0.0))
        attackers_alive = info.get("attackers_alive", info.get("fleet/attackers_alive_end", 0))
        defenders_alive = info.get("defenders_alive", info.get("fleet/defenders_alive_end", 0))
        
        # Use outcome field as single source of truth (consistent with training)
        outcome = info.get("outcome", "unknown")
        
        if outcome in ["captured", "early_success"]:
            wins += 1
            result = c.colorize("✓ WIN", c.BRIGHT_GREEN)
            result_str = "WIN"
        elif outcome == "all_disabled":
            result = c.colorize("✗ LOST (All Disabled)", c.BRIGHT_RED)
            result_str = "LOST"
        elif outcome == "stalled":
            result = c.colorize("⏸ STALLED", c.YELLOW)
            result_str = "STALLED"
        elif outcome == "timeout":
            result = c.colorize("⏱ TIMEOUT", c.YELLOW)
            result_str = "TIMEOUT"
        else:
            result = c.colorize("? UNKNOWN", c.DIM)
            result_str = "UNKNOWN"
        
        if args.verbose:
            # Print action statistics for this episode
            print()
            print(f"  {c.colorize('Action Statistics:', c.BRIGHT_BLUE)}")
            for i, counts in enumerate(action_counts):
                top_actions = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = ", ".join([f"{ACTION_ABBREV.get(a, a[:4])}:{n}" for a, n in top_actions])
                print(f"    A{i} {unit_names[i]}: {top_str}")
            print()
            print(f"  {c.colorize('Result:', c.BRIGHT_BLUE)} {result}  Reward: {total_reward:.2f}  Steps: {step_count}")
            print()
        else:
            # Compact output
            if ep == 0:
                print(f"  {c.colorize('Ep', c.DIM):>5}  {c.colorize('Reward', c.DIM):>10}  {c.colorize('Steps', c.DIM):>8}  {c.colorize('Result', c.DIM):<12}  {c.colorize('Details', c.DIM)}")
                print(f"  {c.colorize('─' * 60, c.DIM)}")
            
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
            "action_counts": action_counts,
        })
        
        # Increment episode counter (only after successful completion)
        ep += 1
    
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
    
    print(f"  {c.colorize('Reward:', c.BRIGHT_BLUE)}")
    print(f"     Mean:   {c.colorize(f'{mean_reward:>8.2f}', c.BRIGHT_GREEN if mean_reward > 0 else c.BRIGHT_RED)} ± {std_reward:.2f}")
    print(f"     Min:    {min_reward:>8.2f}")
    print(f"     Max:    {max_reward:>8.2f}")
    print()
    
    print(f"  {c.colorize('Episodes:', c.BRIGHT_BLUE)}")
    print(f"     Mean length:  {mean_length:.0f} steps")
    print(f"     Win rate:     {c.colorize(f'{win_rate*100:.0f}%', c.BRIGHT_GREEN if win_rate > 0.5 else c.YELLOW)} ({wins}/{args.episodes})")
    print()
    
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
            json.dump(results, f, indent=2, default=str)
        print_success(f"Results saved to {args.save_results}")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
