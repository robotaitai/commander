#!/usr/bin/env python3
"""Utilities for managing training runs with proper naming and organization."""

import json
import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Two-word run name generator (adjective + noun, like Docker containers)
ADJECTIVES = [
    "swift", "brave", "calm", "eager", "fierce", "gentle", "happy", "keen",
    "lucky", "noble", "proud", "quick", "sharp", "smart", "bold", "cool",
    "dark", "epic", "fast", "gold", "iron", "jade", "kind", "late",
    "mega", "neon", "pink", "pure", "red", "sage", "teal", "ultra",
    "vast", "warm", "wild", "zen", "blue", "cyan", "gray", "mint",
]

NOUNS = [
    "falcon", "tiger", "eagle", "wolf", "hawk", "lion", "bear", "fox",
    "dragon", "phoenix", "raven", "shark", "cobra", "viper", "panther",
    "thunder", "storm", "blaze", "frost", "shadow", "light", "spark",
    "comet", "nebula", "quasar", "nova", "pulse", "wave", "ray", "beam",
    "knight", "scout", "ranger", "pilot", "agent", "squad", "force", "unit",
]


def generate_run_name() -> str:
    """Generate a unique run name like 'swift-falcon-20260121-143052'."""
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{adj}-{noun}-{timestamp}"


def get_runs_dir() -> Path:
    """Get the runs directory (create if needed)."""
    # Find project root (where configs/ is located)
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "configs").is_dir():
            break
        current = current.parent
    
    runs_dir = current / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def create_run_dir(run_name: Optional[str] = None) -> Path:
    """Create a new run directory with all needed subdirectories.
    
    If run_name is None, generates a name like 'swift-falcon-20260121-143052'.
    If run_name is provided, appends timestamp like 'my-experiment-20260121-143052'.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_name is None:
        run_name = generate_run_name()
    else:
        # Append timestamp to custom names for uniqueness
        run_name = f"{run_name}-{timestamp}"
    
    runs_dir = get_runs_dir()
    run_dir = runs_dir / run_name
    
    # Create subdirectories
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_run_configs(run_dir: Path) -> None:
    """Copy all YAML config files to the run directory."""
    from mission_gym.config import get_config_dir
    
    config_dir = get_config_dir()
    dest_dir = run_dir / "configs"
    
    for yaml_file in config_dir.glob("*.yaml"):
        shutil.copy2(yaml_file, dest_dir / yaml_file.name)


def save_run_metadata(
    run_dir: Path,
    args: Dict,
    extra: Optional[Dict] = None,
) -> None:
    """Save run metadata (args, system info, etc.)."""
    metadata = {
        "run_name": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "args": args,
        "system": get_system_info(),
    }
    
    if extra:
        metadata.update(extra)
    
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def get_system_info() -> Dict:
    """Get system information including GPU stats."""
    info = {
        "python_version": os.popen("python --version 2>&1").read().strip(),
        "hostname": os.popen("hostname").read().strip(),
    }
    
    # Get nvidia-smi info
    gpu_info = get_nvidia_smi_info()
    if gpu_info:
        info["gpu"] = gpu_info
    
    return info


def get_nvidia_smi_info() -> Optional[Dict]:
    """Get NVIDIA GPU information using nvidia-smi."""
    try:
        # Get basic GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode != 0:
            return None
        
        lines = result.stdout.strip().split("\n")
        gpus = []
        
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 9:
                gpus.append({
                    "index": i,
                    "name": parts[0],
                    "memory_total_mb": int(float(parts[1])) if parts[1] != "[N/A]" else None,
                    "memory_used_mb": int(float(parts[2])) if parts[2] != "[N/A]" else None,
                    "memory_free_mb": int(float(parts[3])) if parts[3] != "[N/A]" else None,
                    "gpu_util_pct": int(float(parts[4])) if parts[4] != "[N/A]" else None,
                    "mem_util_pct": int(float(parts[5])) if parts[5] != "[N/A]" else None,
                    "temperature_c": int(float(parts[6])) if parts[6] != "[N/A]" else None,
                    "power_draw_w": float(parts[7]) if parts[7] != "[N/A]" else None,
                    "power_limit_w": float(parts[8]) if parts[8] != "[N/A]" else None,
                })
        
        # Get driver version
        driver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        driver_version = driver_result.stdout.strip().split("\n")[0] if driver_result.returncode == 0 else None
        
        return {
            "driver_version": driver_version,
            "gpus": gpus,
            "timestamp": datetime.now().isoformat(),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    except Exception:
        return None


def save_rewards_history(
    run_dir: Path,
    rewards: List[float],
    timesteps: List[int] = None,
    episode_lengths: List[int] = None,
) -> None:
    """Save rewards history with detailed statistics to JSON file."""
    import numpy as np
    
    rewards_arr = np.array(rewards) if rewards else np.array([0])
    
    # Calculate statistics
    stats = {
        "total_episodes": len(rewards),
        "total_timesteps": timesteps[-1] if timesteps else 0,
        "mean_reward": float(np.mean(rewards_arr)) if len(rewards) > 0 else 0,
        "std_reward": float(np.std(rewards_arr)) if len(rewards) > 0 else 0,
        "min_reward": float(np.min(rewards_arr)) if len(rewards) > 0 else 0,
        "max_reward": float(np.max(rewards_arr)) if len(rewards) > 0 else 0,
        "final_mean_reward": float(np.mean(rewards_arr[-100:])) if len(rewards) > 0 else 0,
        "final_mean_reward_10": float(np.mean(rewards_arr[-10:])) if len(rewards) > 0 else 0,
    }
    
    # Reward trend (improvement from first 10% to last 10%)
    if len(rewards) >= 20:
        n = max(len(rewards) // 10, 1)
        early_mean = float(np.mean(rewards_arr[:n]))
        late_mean = float(np.mean(rewards_arr[-n:]))
        stats["reward_improvement"] = late_mean - early_mean
        stats["early_mean"] = early_mean
        stats["late_mean"] = late_mean
    
    # Episode length stats
    if episode_lengths:
        lengths_arr = np.array(episode_lengths)
        stats["mean_length"] = float(np.mean(lengths_arr))
        stats["min_length"] = int(np.min(lengths_arr))
        stats["max_length"] = int(np.max(lengths_arr))
    
    history = {
        "rewards": [float(r) for r in rewards],
        "timesteps": timesteps or list(range(len(rewards))),
        "episode_lengths": episode_lengths or [],
        "statistics": stats,
        "saved_at": datetime.now().isoformat(),
    }
    
    with open(run_dir / "rewards_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Also save a simple summary text file
    summary = f"""Training Run Summary
====================
Run: {run_dir.name}
Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Episodes: {stats['total_episodes']:,}
Timesteps: {stats['total_timesteps']:,}

Reward Statistics:
  Mean:      {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}
  Min:       {stats['min_reward']:.2f}
  Max:       {stats['max_reward']:.2f}
  Final 100: {stats['final_mean_reward']:.2f}
  Final 10:  {stats['final_mean_reward_10']:.2f}
"""
    if "reward_improvement" in stats:
        summary += f"""
Improvement:
  Early (first 10%):  {stats['early_mean']:.2f}
  Late (last 10%):    {stats['late_mean']:.2f}
  Improvement:        {stats['reward_improvement']:+.2f}
"""
    
    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)


# Terminal colors for pretty output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    
    @classmethod
    def colorize(cls, text: str, *colors) -> str:
        """Apply colors to text."""
        color_str = "".join(colors)
        return f"{color_str}{text}{cls.RESET}"


def print_banner(run_name: str) -> None:
    """Print a colorful banner for the training run."""
    c = Colors
    width = 70
    
    print()
    print(c.colorize("═" * width, c.BRIGHT_CYAN))
    print(c.colorize("  🎮  ", c.BRIGHT_YELLOW) + c.colorize("MISSION GYM - PPO TRAINING", c.BOLD, c.BRIGHT_CYAN))
    print(c.colorize("═" * width, c.BRIGHT_CYAN))
    print()
    print(c.colorize("  📁 Run: ", c.BRIGHT_BLUE) + c.colorize(run_name, c.BOLD, c.BRIGHT_GREEN))
    print()


def print_gpu_status() -> None:
    """Print GPU status with colors."""
    c = Colors
    gpu_info = get_nvidia_smi_info()
    
    if not gpu_info:
        print(c.colorize("  ⚠️  No NVIDIA GPU detected", c.YELLOW))
        return
    
    print(c.colorize("  🖥️  GPU Status:", c.BOLD, c.BRIGHT_CYAN))
    print()
    
    for gpu in gpu_info.get("gpus", []):
        name = gpu.get("name", "Unknown")
        mem_used = gpu.get("memory_used_mb", 0)
        mem_total = gpu.get("memory_total_mb", 1)
        gpu_util = gpu.get("gpu_util_pct", 0)
        mem_util = gpu.get("mem_util_pct", 0)
        temp = gpu.get("temperature_c", 0)
        power_draw = gpu.get("power_draw_w", 0)
        power_limit = gpu.get("power_limit_w", 1)
        
        # Memory bar
        mem_pct = (mem_used / mem_total * 100) if mem_total else 0
        bar_len = 20
        filled = int(bar_len * mem_pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        # Color based on utilization
        if mem_pct > 90:
            bar_color = c.BRIGHT_RED
        elif mem_pct > 70:
            bar_color = c.BRIGHT_YELLOW
        else:
            bar_color = c.BRIGHT_GREEN
        
        # Temperature color
        if temp > 80:
            temp_color = c.BRIGHT_RED
        elif temp > 70:
            temp_color = c.BRIGHT_YELLOW
        else:
            temp_color = c.BRIGHT_GREEN
        
        print(f"     {c.colorize(name, c.BOLD, c.WHITE)}")
        print(f"     Memory:  {c.colorize(bar, bar_color)} {mem_used:,}/{mem_total:,} MB ({mem_pct:.1f}%)")
        print(f"     GPU:     {c.colorize(f'{gpu_util}%', c.BRIGHT_CYAN)}  |  Mem: {c.colorize(f'{mem_util}%', c.BRIGHT_CYAN)}  |  Temp: {c.colorize(f'{temp}°C', temp_color)}")
        if power_draw and power_limit:
            power_pct = (power_draw / power_limit * 100)
            print(f"     Power:   {power_draw:.1f}W / {power_limit:.1f}W ({power_pct:.0f}%)")
        print()


def print_step(step_num: int, message: str, success: bool = True) -> None:
    """Print a step with colored output."""
    c = Colors
    status = c.colorize("✓", c.BRIGHT_GREEN) if success else c.colorize("✗", c.BRIGHT_RED)
    step_str = c.colorize(f"[{step_num}]", c.BRIGHT_BLUE, c.BOLD)
    print(f"  {step_str} {message}... {status}")


def print_info(message: str) -> None:
    """Print an info message."""
    c = Colors
    print(f"     {c.colorize('→', c.CYAN)} {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    c = Colors
    print(f"     {c.colorize('⚠', c.YELLOW)} {c.colorize(message, c.YELLOW)}")


def print_error(message: str) -> None:
    """Print an error message."""
    c = Colors
    print(f"     {c.colorize('✗', c.BRIGHT_RED)} {c.colorize(message, c.BRIGHT_RED)}")


def print_success(message: str) -> None:
    """Print a success message."""
    c = Colors
    print(f"     {c.colorize('✓', c.BRIGHT_GREEN)} {c.colorize(message, c.BRIGHT_GREEN)}")


def print_divider() -> None:
    """Print a divider line."""
    c = Colors
    print(c.colorize("  " + "─" * 66, c.DIM))
