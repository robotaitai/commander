#!/usr/bin/env python3
"""Utilities for managing training runs with proper naming and organization."""

import hashlib
import json
import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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


def load_eval_seeds() -> List[int]:
    """
    Load held-out evaluation seeds from eval_seeds.txt.
    
    Returns:
        List of seed integers for consistent evaluation
    """
    from mission_gym.config import get_config_dir
    
    eval_seeds_file = get_config_dir() / "eval_seeds.txt"
    
    if not eval_seeds_file.exists():
        # Return default seeds if file doesn't exist
        return [42, 123, 777, 1337, 2024, 314159, 9999, 54321, 11111, 88888]
    
    seeds = []
    with open(eval_seeds_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                try:
                    seeds.append(int(line))
                except ValueError:
                    pass  # Skip invalid lines
    
    return seeds if seeds else [42, 123, 777]  # Fallback if file is empty


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


# ============================================================================
# Lineage Tracking and Compatibility Checking
# ============================================================================

def get_git_commit_hash() -> str:
    """Get current git commit hash, or 'unknown' if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent,  # Go to repo root
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "unknown"


def compute_config_hash(config_dir: Path) -> str:
    """Compute SHA256 hash of all config files in directory."""
    hasher = hashlib.sha256()
    
    # Sort files for deterministic hash
    yaml_files = sorted(config_dir.glob("*.yaml"))
    
    for yaml_file in yaml_files:
        # Include filename in hash
        hasher.update(yaml_file.name.encode())
        # Include file contents
        with open(yaml_file, "rb") as f:
            hasher.update(f.read())
    
    return hasher.hexdigest()


def obs_space_signature(space) -> Dict[str, Any]:
    """Extract observation space signature for compatibility checking."""
    from gymnasium import spaces
    
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": [int(x) for x in space.shape],
            "dtype": str(space.dtype),
            "low": float(space.low.flat[0]) if space.low.size > 0 else None,
            "high": float(space.high.flat[0]) if space.high.size > 0 else None,
        }
    elif isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {k: obs_space_signature(v) for k, v in space.spaces.items()},
        }
    elif isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": [int(x) for x in space.nvec],
        }
    else:
        return {
            "type": type(space).__name__,
            "shape": getattr(space, "shape", None),
        }


def action_space_signature(space) -> Dict[str, Any]:
    """Extract action space signature for compatibility checking."""
    from gymnasium import spaces
    
    if isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
        }
    elif isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": [int(x) for x in space.nvec],
        }
    elif isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": [int(x) for x in space.shape],
            "dtype": str(space.dtype),
        }
    else:
        return {
            "type": type(space).__name__,
        }


def get_root_ancestor_name(checkpoint_path: str) -> Optional[str]:
    """
    Extract the root ancestor name from a checkpoint's lineage tree.
    
    Traverses the lineage back to the root (no parent) and returns its name.
    This is used for family-based naming (e.g., all descendants of 'warm-panther').
    
    Returns:
        The root ancestor's run name, or None if not found.
    """
    if not checkpoint_path:
        return None
    
    parent_path = Path(checkpoint_path)
    
    # Extract run name from checkpoint path
    if "runs" not in parent_path.parts:
        return None
    
    runs_idx = parent_path.parts.index("runs")
    if runs_idx + 1 >= len(parent_path.parts):
        return None
    
    current_run_name = parent_path.parts[runs_idx + 1]
    
    # Traverse lineage to find root
    max_depth = 10  # Prevent infinite loops
    depth = 0
    
    while depth < max_depth:
        lineage_path = get_runs_dir() / current_run_name / "lineage.json"
        
        if not lineage_path.exists():
            # No lineage = this is the root
            return current_run_name
        
        try:
            with open(lineage_path, "r") as f:
                lineage = json.load(f)
            
            parent_name = lineage.get("parent_run_name")
            if not parent_name:
                # No parent = this is the root
                return current_run_name
            
            # Move up the tree
            current_run_name = parent_name
            depth += 1
        except Exception:
            return current_run_name
    
    return current_run_name


def save_lineage(
    run_dir: Path,
    parent_checkpoint: Optional[str] = None,
    branch_name: Optional[str] = None,
    notes: Optional[str] = None,
    obs_space = None,
    action_space = None,
) -> None:
    """Save lineage information for policy branching."""
    lineage = {
        "created_at": datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "config_hash": compute_config_hash(run_dir / "configs"),
    }
    
    # Add parent information if loading from checkpoint
    if parent_checkpoint:
        parent_path = Path(parent_checkpoint)
        lineage["parent_checkpoint_path"] = str(parent_checkpoint)
        
        # Try to extract parent run info from path
        if "runs" in parent_path.parts:
            runs_idx = parent_path.parts.index("runs")
            if runs_idx + 1 < len(parent_path.parts):
                parent_run_name = parent_path.parts[runs_idx + 1]
                lineage["parent_run_name"] = parent_run_name
                lineage["parent_run_dir"] = str(get_runs_dir() / parent_run_name)
                
                # Find and save root ancestor name for family tracking
                root_ancestor = get_root_ancestor_name(parent_checkpoint)
                if root_ancestor:
                    lineage["root_ancestor"] = root_ancestor
                
                # Try to load parent's lineage if it exists
                parent_lineage_path = get_runs_dir() / parent_run_name / "lineage.json"
                if parent_lineage_path.exists():
                    with open(parent_lineage_path, "r") as f:
                        parent_lineage = json.load(f)
                        lineage["parent_lineage"] = parent_lineage
    
    # Add branching metadata
    if branch_name:
        lineage["branch_name"] = branch_name
    if notes:
        lineage["notes"] = notes
    
    # Add space signatures
    if obs_space:
        lineage["obs_space_signature"] = obs_space_signature(obs_space)
    if action_space:
        lineage["action_space_signature"] = action_space_signature(action_space)
    
    # Save to lineage.json
    with open(run_dir / "lineage.json", "w") as f:
        json.dump(lineage, f, indent=2, default=str)
    
    # Also merge into run_metadata.json if it exists
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata["lineage"] = lineage
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


def normalize_space_signature(sig: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize space signature by converting string values to appropriate types.
    Handles legacy lineage files that saved nvec/shape as strings.
    """
    sig = sig.copy()
    
    # Convert nvec strings to ints (for MultiDiscrete)
    if "nvec" in sig and isinstance(sig["nvec"], list):
        sig["nvec"] = [int(x) if isinstance(x, str) else x for x in sig["nvec"]]
    
    # Convert shape strings to ints (for Box)
    if "shape" in sig and isinstance(sig["shape"], list):
        sig["shape"] = [int(x) if isinstance(x, str) else x for x in sig["shape"]]
    
    # Recursively handle Dict spaces
    if "spaces" in sig and isinstance(sig["spaces"], dict):
        sig["spaces"] = {k: normalize_space_signature(v) for k, v in sig["spaces"].items()}
    
    return sig


def check_checkpoint_compatibility(
    checkpoint_path: str,
    current_obs_space,
    current_action_space,
) -> Tuple[bool, Optional[str]]:
    """Check if checkpoint is compatible with current environment.
    
    Returns:
        (is_compatible, error_message)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Auto-append .zip if not present and file doesn't exist
    if not checkpoint_path.exists() and not str(checkpoint_path).endswith('.zip'):
        checkpoint_with_zip = Path(str(checkpoint_path) + '.zip')
        if checkpoint_with_zip.exists():
            checkpoint_path = checkpoint_with_zip
    
    # Try to find parent run's lineage
    if "runs" in checkpoint_path.parts:
        runs_idx = checkpoint_path.parts.index("runs")
        if runs_idx + 1 < len(checkpoint_path.parts):
            parent_run_name = checkpoint_path.parts[runs_idx + 1]
            parent_lineage_path = get_runs_dir() / parent_run_name / "lineage.json"
            
            if parent_lineage_path.exists():
                with open(parent_lineage_path, "r") as f:
                    parent_lineage = json.load(f)
                
                # Compare observation space
                if "obs_space_signature" in parent_lineage:
                    parent_obs_sig = normalize_space_signature(parent_lineage["obs_space_signature"])
                    current_obs_sig = normalize_space_signature(obs_space_signature(current_obs_space))
                    
                    if parent_obs_sig != current_obs_sig:
                        return False, (
                            f"Observation space mismatch!\n"
                            f"  Parent: {json.dumps(parent_obs_sig, indent=2)}\n"
                            f"  Current: {json.dumps(current_obs_sig, indent=2)}\n"
                            f"This usually means you changed:\n"
                            f"  - Number of units in scenario\n"
                            f"  - Observation features (vec_dim)\n"
                            f"  - From Dict to Box or vice versa\n"
                        )
                
                # Compare action space
                if "action_space_signature" in parent_lineage:
                    parent_act_sig = normalize_space_signature(parent_lineage["action_space_signature"])
                    current_act_sig = normalize_space_signature(action_space_signature(current_action_space))
                    
                    if parent_act_sig != current_act_sig:
                        return False, (
                            f"Action space mismatch!\n"
                            f"  Parent: {json.dumps(parent_act_sig, indent=2)}\n"
                            f"  Current: {json.dumps(current_act_sig, indent=2)}\n"
                            f"This usually means you changed:\n"
                            f"  - Number of units\n"
                            f"  - Number of actions per unit\n"
                        )
    
    # If no lineage found or all checks passed
    return True, None


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


def build_lineage_tree_html(runs_data: List[Dict], run_name: Optional[str] = None) -> str:
    """
    Build HTML for lineage tree visualization.
    
    Args:
        runs_data: List of run data dictionaries
        run_name: If provided, highlight this run in the tree
    
    Returns:
        HTML string for the lineage tree
    """
    if not runs_data:
        return "<p style='color: var(--text-secondary);'>No lineage data available</p>"
    
    # Build parent -> children mapping
    runs_by_name = {r["name"]: r for r in runs_data}
    children = {}
    for run in runs_data:
        parent = run.get("parent")
        if parent:
            if parent not in children:
                children[parent] = []
            children[parent].append(run["name"])
    
    # Find roots (runs with no parent or parent doesn't exist)
    run_names = {r["name"] for r in runs_data}
    roots = []
    for run in runs_data:
        parent = run.get("parent")
        if not parent or parent not in run_names:
            roots.append(run["name"])
    
    # Recursively build tree HTML
    def build_tree_node(node_name: str, is_last: bool = True, prefix: str = "") -> str:
        run = runs_by_name.get(node_name)
        if not run:
            return ""
        
        # Highlight current run
        highlight_class = " current-run" if node_name == run_name else ""
        
        # Format metadata
        created = run.get("created", "")
        if created:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                created_str = created[:16] if len(created) > 16 else created
        else:
            created_str = "unknown"
        
        timesteps = run.get("timesteps", 0)
        if timesteps >= 1_000_000:
            timesteps_str = f"{timesteps/1_000_000:.1f}M"
        elif timesteps >= 1_000:
            timesteps_str = f"{timesteps/1_000:.0f}K"
        else:
            timesteps_str = f"{timesteps}"
        
        # Build node
        connector = "└── " if is_last else "├── "
        html = f'<div class="tree-node{highlight_class}">'
        html += f'<span class="tree-line">{prefix}{connector}</span>'
        html += f'<span class="run-name">{node_name}</span>'
        html += f'<span class="run-meta">[{created_str}, {timesteps_str} steps]</span>'
        
        # Add notes if present
        notes = run.get("lineage", {}).get("notes")
        if notes:
            child_prefix = prefix + ("    " if is_last else "│   ")
            html += f'<div class="tree-notes">{child_prefix}   ➜ {notes}</div>'
        
        html += '</div>'
        
        # Build children
        child_names = children.get(node_name, [])
        for i, child_name in enumerate(child_names):
            is_last_child = (i == len(child_names) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            html += build_tree_node(child_name, is_last_child, child_prefix)
        
        return html
    
    # Build all trees
    tree_html = ""
    for root in roots:
        tree_html += build_tree_node(root)
    
    return f'<div class="lineage-tree">{tree_html}</div>'


def generate_unified_dashboard(lineage_filter: Optional[str] = None) -> Path:
    """
    Generate a unified dashboard HTML that allows selecting between runs.
    
    Args:
        lineage_filter: If provided, only show runs in this lineage tree.
                       Can be a run name (shows that run + all descendants).
                       Use "active" to show only the most recent lineage tree.
    
    Returns the path to the generated dashboard.
    """
    runs_dir = get_runs_dir()
    dashboard_path = runs_dir / "dashboard.html"
    
    # Collect all runs with their metadata
    all_runs_data = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        dashboard_file = run_dir / "dashboard.html"
        metadata_file = run_dir / "run_metadata.json"
        lineage_file = run_dir / "lineage.json"
        
        if not dashboard_file.exists():
            continue  # Skip runs without dashboards
        
        # Load metadata if available
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        # Load lineage if available
        lineage = {}
        parent_run = None
        if lineage_file.exists():
            try:
                with open(lineage_file) as f:
                    lineage = json.load(f)
                    parent_run = lineage.get("parent_run_name")
            except Exception:
                pass
        
        # Get dashboard modification time to find the most recently updated (active) run
        dashboard_mtime = dashboard_file.stat().st_mtime
        
        all_runs_data.append({
            "name": run_dir.name,
            "path": str(dashboard_file.relative_to(runs_dir)),
            "created": metadata.get("created_at", ""),
            "timesteps": metadata.get("args", {}).get("timesteps", 0),
            "mtime": dashboard_mtime,  # For sorting by most recently updated
            "parent": parent_run,
            "lineage": lineage,
        })
    
    # Sort by dashboard modification time (most recently updated first = active run)
    all_runs_data.sort(key=lambda x: x["mtime"], reverse=True)
    
    # Apply lineage filtering
    if lineage_filter:
        if lineage_filter == "active":
            # Find the most recent run and show its entire lineage tree
            if all_runs_data:
                root_run = all_runs_data[0]["name"]
                # Find root of this lineage tree
                while all_runs_data[0].get("parent"):
                    parent = all_runs_data[0]["parent"]
                    parent_data = next((r for r in all_runs_data if r["name"] == parent), None)
                    if parent_data:
                        root_run = parent
                        all_runs_data.insert(0, all_runs_data.pop(all_runs_data.index(parent_data)))
                    else:
                        break
                lineage_filter = root_run
        
        # Build lineage tree: include the root and all descendants
        def is_in_lineage_tree(run_data, root_name):
            # Check if this run is the root or descendant of root
            current = run_data["name"]
            visited = set()
            while current:
                if current == root_name:
                    return True
                if current in visited:  # Prevent infinite loops
                    break
                visited.add(current)
                # Find parent
                run_info = next((r for r in all_runs_data if r["name"] == current), None)
                if not run_info:
                    break
                current = run_info.get("parent")
            return False
        
        runs_data = [r for r in all_runs_data if is_in_lineage_tree(r, lineage_filter)]
    else:
        runs_data = all_runs_data
    
    # Build run options HTML with lineage indicators
    options_html = ""
    for i, run in enumerate(runs_data):
        selected = "selected" if i == 0 else ""
        created = run["created"][:19].replace("T", " ") if run["created"] else ""
        steps = f"{run['timesteps']:,}" if run["timesteps"] else "?"
        
        # Add lineage indicator
        prefix = ""
        if run.get("parent"):
            # Count depth in lineage tree
            depth = 0
            current = run["name"]
            visited = set()
            while current and current not in visited:
                visited.add(current)
                parent_data = next((r for r in runs_data if r["name"] == current), None)
                if parent_data and parent_data.get("parent"):
                    depth += 1
                    current = parent_data["parent"]
                else:
                    break
            prefix = "  " * depth + "↳ "
        
        options_html += f'<option value="{run["path"]}" {selected}>{prefix}{run["name"]} ({created}, {steps} steps)</option>\n'
    
    # Default to first run or empty
    default_src = runs_data[0]["path"] if runs_data else ""
    
    # Build lineage tree HTML
    lineage_tree_html = build_lineage_tree_html(runs_data, runs_data[0]["name"] if runs_data else None)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <!-- No meta refresh - using JS to refresh iframe without scroll jump -->
    <title>Mission Gym - All Runs Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0a0e14;
            --bg-panel: #111820;
            --bg-card: #171f2a;
            --border: #2a3544;
            --accent-teal: #00d4aa;
            --accent-cyan: #00bfff;
            --text-primary: #e6edf3;
            --text-secondary: #7d8590;
            --glow-teal: rgba(0, 212, 170, 0.3);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
        }}
        
        .header {{
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
        }}
        
        .logo {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent-teal);
            text-shadow: 0 0 20px var(--glow-teal);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .run-selector {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .run-selector label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .run-selector select {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 1rem;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            min-width: 400px;
            cursor: pointer;
        }}
        
        .run-selector select:hover {{
            border-color: var(--accent-teal);
        }}
        
        .run-selector select:focus {{
            outline: none;
            border-color: var(--accent-teal);
            box-shadow: 0 0 0 2px var(--glow-teal);
        }}
        
        .run-count {{
            background: var(--bg-card);
            border: 1px solid var(--accent-teal);
            border-radius: 20px;
            padding: 0.25rem 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--accent-teal);
        }}
        
        .lineage-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 1rem;
            color: var(--text-primary);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .lineage-btn:hover {{
            border-color: var(--accent-teal);
            color: var(--accent-teal);
        }}
        
        .main-container {{
            display: flex;
            margin-top: 60px;
            height: calc(100vh - 60px);
        }}
        
        .sidebar {{
            width: 400px;
            background: var(--bg-panel);
            border-right: 1px solid var(--border);
            overflow-y: auto;
            display: none;
        }}
        
        .sidebar.show {{
            display: block;
        }}
        
        .sidebar-header {{
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .sidebar-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--accent-teal);
        }}
        
        .close-sidebar {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.2rem;
            cursor: pointer;
            padding: 0.25rem;
        }}
        
        .close-sidebar:hover {{
            color: var(--accent-teal);
        }}
        
        .sidebar-content {{
            padding: 1rem;
        }}
        
        .dashboard-frame {{
            border: none;
            flex: 1;
            height: 100%;
        }}
        
        .lineage-tree {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.6;
        }}
        
        .tree-node {{
            margin: 0.25rem 0;
        }}
        
        .tree-node.current-run {{
            background: rgba(0, 212, 170, 0.1);
            border-left: 3px solid var(--accent-teal);
            padding-left: 0.5rem;
            margin-left: -0.5rem;
        }}
        
        .tree-line {{
            color: var(--text-secondary);
        }}
        
        .run-name {{
            color: var(--text-primary);
            font-weight: 500;
        }}
        
        .run-meta {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-left: 0.5rem;
        }}
        
        .tree-notes {{
            color: var(--accent-cyan);
            font-size: 0.8rem;
            margin-top: 0.25rem;
            font-style: italic;
        }}
        
        .no-runs {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-secondary);
        }}
        
        .no-runs-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
        }}
        
        .no-runs-text {{
            font-size: 1.2rem;
        }}
        
        .no-runs-hint {{
            font-size: 0.9rem;
            margin-top: 0.5rem;
            color: var(--text-secondary);
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">
            <span>🎮</span>
            <span>MISSION GYM</span>
        </div>
        <div class="run-selector">
            <button class="lineage-btn" onclick="toggleSidebar()">🌳 Lineage Tree</button>
            <label for="run-select">Select Run:</label>
            <select id="run-select" onchange="loadRun(this.value)">
                {options_html}
            </select>
            <span class="run-count">{len(runs_data)} runs</span>
        </div>
    </header>
    
    <div class="main-container">
        <div class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <span class="sidebar-title">🌳 Policy Lineage Tree</span>
                <button class="close-sidebar" onclick="toggleSidebar()">✕</button>
            </div>
            <div class="sidebar-content">
                {lineage_tree_html}
            </div>
        </div>
        {f"<iframe class='dashboard-frame' id='dashboard-frame' src='{default_src}'></iframe>" if runs_data else """
        <div class="no-runs">
            <div class="no-runs-icon">&#128640;</div>
            <div class="no-runs-text">No training runs found</div>
            <div class="no-runs-hint">Start a training run with: python -m mission_gym.scripts.train_ppo</div>
        </div>
        """}
    </div>
    
    <script>
        function loadRun(path) {{
            const frame = document.getElementById('dashboard-frame');
            if (frame) {{
                frame.src = path;
            }}
        }}
        
        function toggleSidebar() {{
            const sidebar = document.getElementById('sidebar');
            if (sidebar) {{
                sidebar.classList.toggle('show');
                // Save state
                localStorage.setItem('sidebarOpen', sidebar.classList.contains('show'));
            }}
        }}
        
        // NO auto-refresh of parent page!
        // Each run's dashboard inside the iframe has its own refresh.
        // This prevents scroll jumping and double-refreshes.
        
        // Store the currently selected run in localStorage
        // Default to first run (most recently updated = active training run)
        const select = document.getElementById('run-select');
        if (select && select.options.length > 0) {{
            const saved = localStorage.getItem('selectedRun');
            const defaultRun = select.options[0].value;  // Most recently updated dashboard
            
            // Check if saved run still exists and is valid
            let runToLoad = defaultRun;
            if (saved) {{
                // Check if saved run exists in options
                let savedExists = false;
                for (let option of select.options) {{
                    if (option.value === saved) {{
                        savedExists = true;
                        break;
                    }}
                }}
                // Use saved run if it exists, otherwise use default (active run)
                if (savedExists) {{
                    runToLoad = saved;
                }}
            }}
            
            select.value = runToLoad;
            loadRun(runToLoad);
            
            // Save user's manual selection
            select.addEventListener('change', function() {{
                localStorage.setItem('selectedRun', this.value);
            }});
        }}
        
        // Restore sidebar state
        const sidebarOpen = localStorage.getItem('sidebarOpen');
        if (sidebarOpen === 'true') {{
            const sidebar = document.getElementById('sidebar');
            if (sidebar) {{
                sidebar.classList.add('show');
            }}
        }}
    </script>
</body>
</html>'''
    
    with open(dashboard_path, "w") as f:
        f.write(html)
    
    return dashboard_path


def update_unified_dashboard() -> None:
    """Update the unified dashboard (call this after creating a new run)."""
    try:
        generate_unified_dashboard()
    except Exception:
        pass  # Non-critical, don't fail training
