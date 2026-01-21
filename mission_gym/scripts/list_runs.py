#!/usr/bin/env python3
"""List all training runs with their performance metrics."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def load_run_info(run_dir: Path) -> Optional[Dict]:
    """Load info about a training run."""
    try:
        info = {
            "name": run_dir.name,
            "path": str(run_dir),
        }
        
        # Parse timestamp from name (format: word-word-YYYYMMDD-HHMMSS)
        parts = run_dir.name.split("-")
        if len(parts) >= 4:
            try:
                date_str = f"{parts[-2]}-{parts[-1]}"
                info["timestamp"] = datetime.strptime(date_str, "%Y%m%d-%H%M%S")
            except ValueError:
                info["timestamp"] = datetime.fromtimestamp(run_dir.stat().st_mtime)
        else:
            info["timestamp"] = datetime.fromtimestamp(run_dir.stat().st_mtime)
        
        # Load metadata
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            info["timesteps"] = metadata.get("args", {}).get("timesteps", "?")
            info["n_envs"] = metadata.get("args", {}).get("n_envs", "?")
            info["gpu"] = metadata.get("system", {}).get("gpu", {}).get("gpus", [{}])[0].get("name", "N/A") if metadata.get("system", {}).get("gpu") else "N/A"
        
        # Load rewards history
        rewards_path = run_dir / "rewards_history.json"
        if rewards_path.exists():
            with open(rewards_path) as f:
                rewards_data = json.load(f)
            rewards = rewards_data.get("rewards", [])
            if rewards:
                info["episodes"] = len(rewards)
                info["mean_reward"] = sum(rewards[-100:]) / min(len(rewards), 100)
                info["max_reward"] = max(rewards)
                info["final_reward"] = sum(rewards[-10:]) / min(len(rewards), 10) if rewards else 0
            else:
                info["episodes"] = 0
                info["mean_reward"] = 0
                info["max_reward"] = 0
                info["final_reward"] = 0
        else:
            info["episodes"] = 0
            info["mean_reward"] = 0
            info["max_reward"] = 0
            info["final_reward"] = 0
        
        # Check for model
        info["has_model"] = (run_dir / "final_model.zip").exists() or len(list((run_dir / "checkpoints").glob("*.zip"))) > 0
        
        # Check for dashboard
        info["has_dashboard"] = (run_dir / "dashboard.html").exists()
        
        return info
    except Exception as e:
        return None


def format_duration(seconds: float) -> str:
    """Format duration in human readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    """List all training runs."""
    parser = argparse.ArgumentParser(description="List Mission Gym training runs")
    parser.add_argument(
        "--sort",
        type=str,
        choices=["date", "reward", "episodes", "name"],
        default="date",
        help="Sort by field (default: date)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse sort order",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max runs to show (default: 20)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()
    
    from mission_gym.scripts.run_utils import get_runs_dir, Colors
    c = Colors
    
    runs_dir = get_runs_dir()
    
    # Collect run info
    runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("."):
            info = load_run_info(run_dir)
            if info:
                runs.append(info)
    
    if not runs:
        print(f"\n  {c.colorize('No training runs found.', c.YELLOW)}")
        print(f"  Start a new run with: python -m mission_gym.scripts.train_ppo\n")
        return 0
    
    # Sort
    sort_keys = {
        "date": lambda x: x["timestamp"],
        "reward": lambda x: x.get("mean_reward", 0),
        "episodes": lambda x: x.get("episodes", 0),
        "name": lambda x: x["name"],
    }
    runs.sort(key=sort_keys[args.sort], reverse=not args.reverse if args.sort in ["date", "reward", "episodes"] else args.reverse)
    
    # Limit
    runs = runs[:args.limit]
    
    if args.json:
        # JSON output
        for run in runs:
            run["timestamp"] = run["timestamp"].isoformat()
        print(json.dumps(runs, indent=2))
        return 0
    
    # Pretty print
    print()
    print(c.colorize("═" * 100, c.BRIGHT_CYAN))
    print(c.colorize("  📁 TRAINING RUNS", c.BOLD, c.BRIGHT_CYAN))
    print(c.colorize("═" * 100, c.BRIGHT_CYAN))
    print()
    
    # Header
    header = f"  {'Run Name':<40} {'Date':<12} {'Episodes':>10} {'Mean R':>10} {'Final R':>10} {'Model':>6}"
    print(c.colorize(header, c.DIM))
    print(c.colorize("  " + "─" * 96, c.DIM))
    
    for run in runs:
        name = run["name"][:38]
        date = run["timestamp"].strftime("%Y-%m-%d")
        episodes = run.get("episodes", 0)
        mean_r = run.get("mean_reward", 0)
        final_r = run.get("final_reward", 0)
        has_model = "✓" if run.get("has_model") else "✗"
        
        # Color code rewards
        if mean_r > 50:
            r_color = c.BRIGHT_GREEN
        elif mean_r > 0:
            r_color = c.GREEN
        elif mean_r > -50:
            r_color = c.YELLOW
        else:
            r_color = c.RED
        
        if final_r > 50:
            f_color = c.BRIGHT_GREEN
        elif final_r > 0:
            f_color = c.GREEN
        elif final_r > -50:
            f_color = c.YELLOW
        else:
            f_color = c.RED
        
        model_color = c.BRIGHT_GREEN if run.get("has_model") else c.RED
        
        print(f"  {c.colorize(name, c.BRIGHT_BLUE):<48} {date:<12} {episodes:>10,} {c.colorize(f'{mean_r:>10.1f}', r_color)} {c.colorize(f'{final_r:>10.1f}', f_color)} {c.colorize(has_model, model_color):>14}")
    
    print()
    print(c.colorize("  " + "─" * 96, c.DIM))
    print(f"  {c.colorize(f'Total: {len(runs)} runs', c.DIM)}")
    print()
    
    # Tips
    print(f"  {c.colorize('💡 Tips:', c.BRIGHT_YELLOW)}")
    print(f"     • Evaluate a run: {c.colorize('python -m mission_gym.scripts.evaluate --model <run-name>', c.CYAN)}")
    print(f"     • View dashboard:  {c.colorize('open runs/<run-name>/dashboard.html', c.CYAN)}")
    print(f"     • TensorBoard:     {c.colorize('tensorboard --logdir runs/<run-name>/logs', c.CYAN)}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
