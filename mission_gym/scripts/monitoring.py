#!/usr/bin/env python3
"""
Training monitoring with HTML dashboard generation.

Inspired by physical-ai.news aesthetic - dark theme, teal accents, 
ticker-style displays, and card-based layouts.
"""

import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def get_nvidia_smi_info() -> Optional[Dict]:
    """Get NVIDIA GPU information using nvidia-smi."""
    import subprocess
    try:
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
        
        return {"gpus": gpus, "timestamp": datetime.now().isoformat()}
    except Exception:
        return None


class HTMLMonitorCallback(BaseCallback):
    """
    Callback that generates an HTML dashboard for monitoring training progress.
    
    Features:
    - Cyberpunk-inspired dark theme (physical-ai.news style)
    - Real-time reward component ticker
    - Interactive breakdown charts
    - GPU utilization monitoring
    - Simulation snapshots
    - Configuration viewer
    """
    
    def __init__(
        self,
        html_path: str = "training_dashboard.html",
        update_freq: int = 1000,
        verbose: int = 0,
        run_dir: Optional[Path] = None,
    ):
        super().__init__(verbose)
        self.html_path = Path(html_path)
        self.update_freq = update_freq
        self.run_dir = run_dir
        
        # Training history
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.timesteps_history: List[int] = []
        self.mean_rewards: List[float] = []
        self.eval_rewards: List[float] = []
        self.eval_timesteps: List[int] = []
        
        # Reward component tracking
        self.component_history: Dict[str, List[float]] = {}
        self.component_configs: List[Dict] = []
        self.latest_component_breakdown: List[Dict] = []
        
        # GPU history
        self.gpu_history: List[Dict] = []
        self.gpu_timestamps: List[str] = []
        
        # Episode tracking - per-environment for vectorized envs
        self.n_envs = 1  # Will be updated on first step
        self.current_episode_rewards: List[float] = [0.0]
        self.current_episode_lengths: List[int] = [0]
        self.episodes_completed = 0
        
        # Stats
        self.start_time: Optional[datetime] = None
        self.last_update_timestep = 0
        
        # Stored snapshots from eval callbacks
        self.stored_snapshots: List[Dict] = []
        
        # Episode metrics tracking
        self.episode_metrics_history: List[Dict] = []
        self.wins = 0
        self.total_episodes = 0
        self.latest_metrics: Optional[Dict] = None
        
        # Action log tracking (last 100 timesteps)
        self.action_log: List[Dict] = []  # Each entry: {timestep, actions, rewards}
        
        # Load config YAML files for display
        self.config_yaml = self._load_config_yamls()
    
    def _load_config_yamls(self) -> Dict[str, str]:
        """Load all YAML config files for display."""
        from mission_gym.config import get_config_dir
        
        configs = {}
        
        # Try run-specific configs first, then fall back to main configs
        if self.run_dir:
            config_dir = self.run_dir / "configs"
        else:
            config_dir = get_config_dir()
        
        yaml_files = [
            "world.yaml",
            "scenario.yaml",
            "units_attackers.yaml",
            "units_defenders.yaml",
            "sensors.yaml",
            "engagement.yaml",
            "reward.yaml",
        ]
        
        for filename in yaml_files:
            try:
                filepath = config_dir / filename
                if filepath.exists():
                    with open(filepath, 'r') as f:
                        content = f.read()
                    configs[filename] = content
            except Exception:
                pass
        
        return configs
    
    def update_component_configs(self, configs: List[Dict]):
        """Update reward component configurations."""
        self.component_configs = configs
    
    def record_component_breakdown(self, breakdown: List[Dict]):
        """Record a reward component breakdown."""
        self.latest_component_breakdown = breakdown
        
        # Track history for each component
        for comp in breakdown:
            name = comp.get("name", "unknown")
            value = comp.get("value", 0)
            if name not in self.component_history:
                self.component_history[name] = []
            self.component_history[name].append(value)
            # Keep last 1000 values
            if len(self.component_history[name]) > 1000:
                self.component_history[name] = self.component_history[name][-1000:]
    
    def _update_gpu_history(self) -> Optional[Dict]:
        """Update GPU history and return current stats."""
        gpu_info = get_nvidia_smi_info()
        if gpu_info and gpu_info.get("gpus"):
            self.gpu_history.append(gpu_info)
            self.gpu_timestamps.append(datetime.now().strftime("%H:%M:%S"))
            if len(self.gpu_history) > 60:
                self.gpu_history = self.gpu_history[-60:]
                self.gpu_timestamps = self.gpu_timestamps[-60:]
        return gpu_info
    
    def _on_training_start(self) -> None:
        self.start_time = datetime.now()
        self._generate_html()
    
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0])
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        actions = self.locals.get("actions", None)
        
        # Record action log entry (one entry per step across all envs)
        if actions is not None:
            action_entry = {
                "timestep": self.num_timesteps,
                "actions": actions.tolist() if hasattr(actions, "tolist") else list(actions),
                "rewards": rewards.tolist() if hasattr(rewards, "tolist") else list(rewards),
                "dones": dones.tolist() if hasattr(dones, "tolist") else list(dones),
            }
            self.action_log.append(action_entry)
            # Keep last 100 entries
            if len(self.action_log) > 100:
                self.action_log = self.action_log[-100:]
        
        # Initialize per-env tracking if needed (for vectorized envs)
        n_envs = len(rewards)
        if n_envs != self.n_envs:
            self.n_envs = n_envs
            self.current_episode_rewards = [0.0] * n_envs
            self.current_episode_lengths = [0] * n_envs
        
        for env_idx, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
            self.current_episode_rewards[env_idx] += reward
            self.current_episode_lengths[env_idx] += 1
            
            # Track component breakdown if available
            if "_component_breakdown" in info:
                self.record_component_breakdown(info["_component_breakdown"])
            
            if done:
                # Record completed episode for this specific env
                self.episode_rewards.append(self.current_episode_rewards[env_idx])
                self.episode_lengths.append(self.current_episode_lengths[env_idx])
                self.episodes_completed += 1
                
                # Track episode metrics - ALWAYS append to keep in sync with episode_rewards
                metrics = info.get("episode_metrics", {})
                self.episode_metrics_history.append(metrics)
                
                if metrics:
                    self.latest_metrics = metrics
                    self.total_episodes += 1
                    if metrics.get("win", False):
                        self.wins += 1
                
                # Trim history to last 100 episodes (all lists must stay in sync)
                if len(self.episode_rewards) > 100:
                    self.episode_rewards = self.episode_rewards[-100:]
                    self.episode_lengths = self.episode_lengths[-100:]
                    self.episode_metrics_history = self.episode_metrics_history[-100:]
                
                if self.episodes_completed % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    self.mean_rewards.append(mean_reward)
                    self.timesteps_history.append(self.num_timesteps)
                
                # Reset this env's accumulators
                self.current_episode_rewards[env_idx] = 0.0
                self.current_episode_lengths[env_idx] = 0
        
        if self.num_timesteps - self.last_update_timestep >= self.update_freq:
            self._generate_html()
            self.last_update_timestep = self.num_timesteps
        
        return True
    
    def _on_training_end(self) -> None:
        self._generate_html()
    
    def add_eval_result(self, mean_reward: float, timestep: int, snapshots: List[Dict] = None) -> None:
        """Add evaluation result to history."""
        self.eval_rewards.append(mean_reward)
        self.eval_timesteps.append(timestep)
        if snapshots:
            self.stored_snapshots = snapshots
        self._generate_html()
    
    def _generate_html(self) -> None:
        """Generate the HTML dashboard with physical-ai.news inspired design."""
        elapsed = datetime.now() - self.start_time if self.start_time else None
        elapsed_str = str(elapsed).split('.')[0] if elapsed else "N/A"
        
        fps = self.num_timesteps / elapsed.total_seconds() if elapsed and elapsed.total_seconds() > 0 else 0
        mean_reward_100 = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        mean_length_100 = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        
        # Calculate win rate from actual episode metrics
        if self.total_episodes > 0:
            win_rate = self.wins / self.total_episodes * 100
        elif self.episode_rewards:
            # Fallback: estimate from positive rewards
            positive_episodes = sum(1 for r in self.episode_rewards[-100:] if r > 50)
            win_rate = positive_episodes / min(100, len(self.episode_rewards)) * 100
        else:
            win_rate = 0
        
        # Get GPU info
        gpu_info = self._update_gpu_history()
        
        # Build component ticker data
        component_ticker = self._build_component_ticker()
        
        # Build reward breakdown data
        reward_breakdown = self._build_reward_breakdown()
        
        run_name = self.run_dir.name if self.run_dir else "Training Session"
        
        # Chart data - show all history from step 0
        chart_data = {
            "timesteps": self.timesteps_history,  # Full history from step 0
            "mean_rewards": self.mean_rewards,    # Full history
            "eval_timesteps": self.eval_timesteps,
            "eval_rewards": self.eval_rewards,
            "component_history": self.component_history,  # Full component history
        }
        
        html = self._build_html_page(
            run_name=run_name,
            elapsed_str=elapsed_str,
            fps=fps,
            mean_reward_100=mean_reward_100,
            mean_length_100=mean_length_100,
            win_rate=win_rate,
            gpu_info=gpu_info,
            component_ticker=component_ticker,
            reward_breakdown=reward_breakdown,
            chart_data=chart_data,
        )
        
        self.html_path.write_text(html)
    
    def _build_component_ticker(self) -> List[Dict]:
        """Build ticker data for reward components."""
        ticker_items = []
        
        if self.component_configs:
            for config in self.component_configs:
                name = config.get("name", "unknown")
                icon = config.get("icon", "📊")
                color = config.get("color", "#58a6ff")
                
                # Get recent average
                if name in self.component_history and self.component_history[name]:
                    recent = self.component_history[name][-100:]
                    avg_value = sum(recent) / len(recent) if recent else 0
                    total = sum(self.component_history[name])
                else:
                    avg_value = 0
                    total = 0
                
                ticker_items.append({
                    "name": name.replace("_", " ").title(),
                    "icon": icon,
                    "color": color,
                    "avg": avg_value,
                    "total": total,
                    "trend": "up" if avg_value > 0 else "down" if avg_value < 0 else "neutral",
                })
        
        return ticker_items
    
    def _build_reward_breakdown(self) -> Dict:
        """Build reward breakdown for visualization."""
        breakdown = {
            "components": [],
            "categories": {},
        }
        
        if self.component_configs:
            for config in self.component_configs:
                name = config.get("name", "unknown")
                category = config.get("category", "shaping")
                color = config.get("color", "#58a6ff")
                icon = config.get("icon", "📊")
                description = config.get("description", "")
                
                # Calculate stats
                if name in self.component_history and self.component_history[name]:
                    values = self.component_history[name]
                    total = sum(values)
                    avg = total / len(values) if values else 0
                    non_zero = sum(1 for v in values if v != 0)
                    pct_active = non_zero / len(values) * 100 if values else 0
                else:
                    total = 0
                    avg = 0
                    pct_active = 0
                
                breakdown["components"].append({
                    "name": name,
                    "display_name": name.replace("_", " ").title(),
                    "category": category,
                    "color": color,
                    "icon": icon,
                    "description": description,
                    "total": total,
                    "average": avg,
                    "active_pct": pct_active,
                })
                
                # Aggregate by category
                if category not in breakdown["categories"]:
                    breakdown["categories"][category] = {"total": 0, "count": 0}
                breakdown["categories"][category]["total"] += total
                breakdown["categories"][category]["count"] += 1
        
        return breakdown
    
    def _build_html_page(
        self,
        run_name: str,
        elapsed_str: str,
        fps: float,
        mean_reward_100: float,
        mean_length_100: float,
        win_rate: float,
        gpu_info: Optional[Dict],
        component_ticker: List[Dict],
        reward_breakdown: Dict,
        chart_data: Dict,
    ) -> str:
        """Build the complete HTML page."""
        
        # Build ticker HTML
        ticker_html = self._build_ticker_html(component_ticker)
        
        # Build GPU HTML
        gpu_html = self._build_gpu_html(gpu_info)
        
        # Build reward breakdown HTML
        breakdown_html = self._build_breakdown_html(reward_breakdown)
        
        # Build snapshots HTML
        snapshots_html = self._build_snapshots_html()
        
        # Build config HTML
        config_html = self._build_config_html()
        
        # Build episode metrics panel
        metrics_html = self._build_episode_metrics_html()
        
        # Build recent episodes table
        recent_html = self._build_recent_episodes_html()
        
        # Build quick commands panel
        commands_html = self._build_commands_html(run_name)
        
        # Build action log panel
        action_log_html = self._build_action_log_html()
        
        # Build policy continuation info panel
        continuation_html = self._build_continuation_info_html()
        
        # Build lineage tree panel
        lineage_html = self._build_lineage_html()
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5">
    <title>{run_name} // Mission Gym</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-dark: #0a0e14;
            --bg-panel: #111820;
            --bg-card: #171f2a;
            --bg-hover: #1e2836;
            --border: #2a3544;
            --accent-teal: #00d4aa;
            --accent-cyan: #00bfff;
            --accent-purple: #a855f7;
            --accent-pink: #ec4899;
            --accent-yellow: #facc15;
            --accent-orange: #f97316;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-primary: #e6edf3;
            --text-secondary: #7d8590;
            --text-dim: #484f58;
            --glow-teal: rgba(0, 212, 170, 0.3);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }}
        
        /* Ticker Strip */
        .ticker-strip {{
            background: linear-gradient(90deg, var(--bg-panel) 0%, var(--bg-card) 50%, var(--bg-panel) 100%);
            border-bottom: 1px solid var(--border);
            padding: 0.5rem 0;
            overflow: hidden;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
        }}
        
        .ticker-content {{
            display: flex;
            animation: ticker 30s linear infinite;
            white-space: nowrap;
        }}
        
        @keyframes ticker {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        
        .ticker-item {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        
        .ticker-item .icon {{ font-size: 1rem; }}
        .ticker-item .name {{ color: var(--text-secondary); }}
        .ticker-item .value {{ font-weight: 600; }}
        .ticker-item .value.positive {{ color: var(--success); }}
        .ticker-item .value.negative {{ color: var(--danger); }}
        .ticker-item .value.neutral {{ color: var(--text-dim); }}
        .ticker-item .separator {{ color: var(--border); margin: 0 0.5rem; }}
        
        /* Header */
        .header {{
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 1.5rem 2rem;
            margin-top: 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header-left {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .logo {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent-teal);
            text-shadow: 0 0 20px var(--glow-teal);
        }}
        
        .run-badge {{
            background: var(--bg-card);
            border: 1px solid var(--accent-teal);
            border-radius: 4px;
            padding: 0.25rem 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--accent-teal);
        }}
        
        .header-right {{
            display: flex;
            gap: 1rem;
            align-items: center;
        }}
        
        .status-indicator {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        /* Main Container */
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            transition: all 0.2s;
        }}
        
        .stat-card:hover {{
            border-color: var(--accent-teal);
            box-shadow: 0 0 20px var(--glow-teal);
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        
        .stat-value {{
            font-size: 1.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .stat-value.positive {{ color: var(--success); }}
        .stat-value.warning {{ color: var(--warning); }}
        .stat-value.accent {{ color: var(--accent-teal); }}
        
        /* Grid Layout */
        .grid-2 {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        /* Panel */
        .panel {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border);
            background: var(--bg-panel);
        }}
        
        .panel-title {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .panel-title .icon {{
            color: var(--accent-teal);
        }}
        
        .panel-body {{
            padding: 1.25rem;
        }}
        
        /* Lineage Tree */
        .lineage-tree {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.8;
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
        
        /* Reward Components */
        .reward-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}
        
        .reward-card {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1rem;
            transition: all 0.2s;
        }}
        
        .reward-card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-cyan);
        }}
        
        .reward-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }}
        
        .reward-icon {{
            font-size: 1.25rem;
        }}
        
        .reward-name {{
            font-weight: 500;
            font-size: 0.9rem;
        }}
        
        .reward-info-btn {{
            margin-left: auto;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 0.7rem;
            color: var(--text-secondary);
            transition: all 0.2s;
        }}
        
        .reward-info-btn:hover {{
            background: var(--accent-teal);
            color: var(--bg-dark);
            border-color: var(--accent-teal);
        }}
        
        .reward-description {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.5;
            margin-top: 0.5rem;
            padding: 0 0.5rem;
            border-left: 2px solid var(--border);
        }}
        
        .reward-description.expanded {{
            max-height: 200px;
            margin-bottom: 0.5rem;
        }}
        
        .reward-category {{
            font-size: 0.7rem;
            padding: 0.15rem 0.5rem;
            border-radius: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .reward-category.objective {{ background: rgba(34, 197, 94, 0.2); color: var(--success); }}
        .reward-category.penalty {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
        .reward-category.shaping {{ background: rgba(0, 191, 255, 0.2); color: var(--accent-cyan); }}
        .reward-category.survival {{ background: rgba(249, 115, 22, 0.2); color: var(--accent-orange); }}
        .reward-category.bonus {{ background: rgba(168, 85, 247, 0.2); color: var(--accent-purple); }}
        
        .reward-stats {{
            display: flex;
            justify-content: space-between;
            margin-top: 0.5rem;
        }}
        
        .reward-stat {{
            text-align: center;
        }}
        
        .reward-stat-label {{
            font-size: 0.65rem;
            color: var(--text-dim);
            text-transform: uppercase;
        }}
        
        .reward-stat-value {{
            font-size: 0.9rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .reward-bar {{
            height: 4px;
            background: var(--bg-dark);
            border-radius: 2px;
            margin-top: 0.75rem;
            overflow: hidden;
        }}
        
        .reward-bar-fill {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s;
        }}
        
        /* GPU Panel */
        .gpu-card {{
            background: var(--bg-panel);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}
        
        .gpu-name {{
            font-weight: 600;
            color: var(--accent-cyan);
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
        }}
        
        .gpu-metrics {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
        }}
        
        .gpu-metric {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .metric-label {{
            font-size: 0.7rem;
            color: var(--text-dim);
            width: 40px;
            text-transform: uppercase;
        }}
        
        .progress-bar {{
            flex: 1;
            height: 6px;
            background: var(--bg-dark);
            border-radius: 3px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s;
        }}
        
        .progress-fill.success {{ background: var(--success); }}
        .progress-fill.warning {{ background: var(--warning); }}
        .progress-fill.danger {{ background: var(--danger); }}
        .progress-fill.accent {{ background: var(--accent-teal); }}
        
        .metric-value {{
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            min-width: 60px;
            text-align: right;
        }}
        
        /* Chart */
        canvas {{
            max-height: 250px;
        }}
        
        /* Snapshots */
        .snapshot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }}
        
        .snapshot-card {{
            background: var(--bg-panel);
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        
        .snapshot-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .snapshot-info {{
            padding: 0.5rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .snapshot-header {{
            font-weight: 600;
            color: var(--accent-teal);
            margin-bottom: 0.25rem;
        }}
        
        .snapshot-metrics {{
            display: flex;
            gap: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Config Tabs */
        .config-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .config-tab {{
            padding: 0.5rem 1rem;
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            transition: all 0.2s;
        }}
        
        .config-tab:hover {{
            border-color: var(--accent-teal);
            color: var(--text-primary);
        }}
        
        .config-tab.active {{
            background: var(--accent-teal);
            color: var(--bg-dark);
            border-color: var(--accent-teal);
        }}
        
        .config-content {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 1rem;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            line-height: 1.6;
            white-space: pre-wrap;
            color: var(--text-secondary);
        }}
        
        .config-content::-webkit-scrollbar {{ width: 6px; }}
        .config-content::-webkit-scrollbar-track {{ background: var(--bg-panel); }}
        .config-content::-webkit-scrollbar-thumb {{ background: var(--accent-teal); border-radius: 3px; }}
        
        /* Table */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background: var(--bg-panel);
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        tr:hover td {{
            background: var(--bg-hover);
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: 500;
        }}
        
        .badge-success {{ background: rgba(34, 197, 94, 0.2); color: var(--success); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-dim);
            font-size: 0.8rem;
        }}
        
        .footer a {{
            color: var(--accent-teal);
            text-decoration: none;
        }}
        
        /* Responsive */
        @media (max-width: 1200px) {{
            .stats-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .grid-2 {{ grid-template-columns: 1fr; }}
            .grid-3 {{ grid-template-columns: 1fr; }}
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .header {{ flex-direction: column; gap: 1rem; }}
        }}
    </style>
</head>
<body>
    <!-- Ticker Strip -->
    {ticker_html}
    
    <!-- Header -->
    <header class="header">
        <div class="header-left">
            <span class="logo">MISSION GYM</span>
            <span class="run-badge">{run_name}</span>
        </div>
        <div class="header-right">
            <div class="status-indicator">
                <span class="status-dot"></span>
                <span>TRAINING</span>
            </div>
            <div class="status-indicator" style="color: var(--text-dim);">
                {datetime.now().strftime('%H:%M:%S')}
            </div>
        </div>
    </header>
    
    <!-- Main Content -->
    <div class="container">
        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Timesteps</div>
                <div class="stat-value accent">{self.num_timesteps:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Episodes</div>
                <div class="stat-value">{self.episodes_completed:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Reward (100)</div>
                <div class="stat-value {'positive' if mean_reward_100 > 0 else 'warning'}">{mean_reward_100:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value {'positive' if win_rate > 50 else 'warning'}">{win_rate:.0f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">FPS</div>
                <div class="stat-value">{fps:.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Elapsed</div>
                <div class="stat-value" style="font-size: 1.2rem;">{elapsed_str}</div>
            </div>
        </div>
        
        <!-- Main Grid -->
        <div class="grid-2">
            <!-- Reward Chart -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title"><span class="icon">📈</span> Reward Progress</div>
                </div>
                <div class="panel-body">
                    <canvas id="rewardChart"></canvas>
                </div>
            </div>
            
            <!-- GPU Status -->
            {gpu_html}
        </div>
        
        <!-- Reward Components Breakdown -->
        {breakdown_html}
        
        <!-- Episode Metrics -->
        {metrics_html}
        
        <!-- Recent Episodes -->
        {recent_html}
        
        <!-- Snapshots -->
        {snapshots_html}
        
        <!-- Quick Commands -->
        {commands_html}
        
        <!-- Action Log -->
        {action_log_html}
        
        <!-- Policy Continuation Info -->
        {continuation_html}
        
        <!-- Policy Lineage Tree -->
        {lineage_html}
        
        <!-- Configuration -->
        {config_html}
    </div>
    
    <footer class="footer">
        <p>Mission Gym Training Dashboard • Auto-refreshes every 5s</p>
    </footer>
    
    <script>
        // Config tab switching
        function showConfig(id) {{
            document.querySelectorAll('.config-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.config-tab').forEach(el => el.classList.remove('active'));
            document.getElementById(id).style.display = 'block';
            event.target.classList.add('active');
        }}
        
        // Toggle reward component description
        function toggleRewardInfo(cardId) {{
            const desc = document.getElementById(cardId);
            desc.classList.toggle('expanded');
        }}
        
        // Chart
        const chartData = {json.dumps(chart_data)};
        
        new Chart(document.getElementById('rewardChart'), {{
            type: 'line',
            data: {{
                labels: chartData.timesteps,
                datasets: [{{
                    label: 'Mean Reward (100 ep)',
                    data: chartData.mean_rewards,
                    borderColor: '#00d4aa',
                    backgroundColor: 'rgba(0, 212, 170, 0.1)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2,
                }}, {{
                    label: 'Eval Reward',
                    data: chartData.eval_timesteps.map((t, i) => ({{ x: t, y: chartData.eval_rewards[i] }})),
                    borderColor: '#a855f7',
                    backgroundColor: '#a855f7',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    showLine: false,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ 
                        labels: {{ color: '#7d8590', font: {{ family: "'Space Grotesk'" }} }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Timesteps', color: '#484f58' }},
                        ticks: {{ color: '#484f58' }},
                        grid: {{ color: '#2a3544' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Reward', color: '#484f58' }},
                        ticks: {{ color: '#484f58' }},
                        grid: {{ color: '#2a3544' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>'''
    
    def _build_ticker_html(self, ticker_items: List[Dict]) -> str:
        """Build the scrolling ticker HTML."""
        if not ticker_items:
            return ""
        
        items_html = ""
        for item in ticker_items:
            value_class = "positive" if item["avg"] > 0 else "negative" if item["avg"] < 0 else "neutral"
            items_html += f'''
            <div class="ticker-item">
                <span class="icon">{item["icon"]}</span>
                <span class="name">{item["name"]}</span>
                <span class="value {value_class}">{item["avg"]:+.3f}</span>
                <span class="separator">│</span>
            </div>'''
        
        # Duplicate for infinite scroll
        return f'''
        <div class="ticker-strip">
            <div class="ticker-content">
                {items_html}
                {items_html}
            </div>
        </div>'''
    
    def _build_gpu_html(self, gpu_info: Optional[Dict]) -> str:
        """Build GPU status panel HTML."""
        if not gpu_info or not gpu_info.get("gpus"):
            return '''
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title"><span class="icon">🖥️</span> GPU Status</div>
                </div>
                <div class="panel-body">
                    <p style="color: var(--text-secondary);">No GPU detected</p>
                </div>
            </div>'''
        
        gpus_html = ""
        for gpu in gpu_info["gpus"]:
            name = gpu.get("name", "Unknown")
            gpu_util = gpu.get("gpu_util_pct", 0) or 0
            mem_used = gpu.get("memory_used_mb", 0) or 0
            mem_total = gpu.get("memory_total_mb", 1) or 1
            mem_pct = (mem_used / mem_total * 100) if mem_total else 0
            temp = gpu.get("temperature_c", 0) or 0
            power = gpu.get("power_draw_w", 0) or 0
            power_limit = gpu.get("power_limit_w", 1) or 1
            power_pct = (power / power_limit * 100) if power_limit else 0
            
            gpu_class = "danger" if gpu_util > 90 else "warning" if gpu_util > 70 else "success"
            mem_class = "danger" if mem_pct > 90 else "warning" if mem_pct > 70 else "success"
            temp_class = "danger" if temp > 80 else "warning" if temp > 70 else "success"
            
            gpus_html += f'''
            <div class="gpu-card">
                <div class="gpu-name">{name}</div>
                <div class="gpu-metrics">
                    <div class="gpu-metric">
                        <span class="metric-label">GPU</span>
                        <div class="progress-bar"><div class="progress-fill {gpu_class}" style="width: {gpu_util}%"></div></div>
                        <span class="metric-value">{gpu_util}%</span>
                    </div>
                    <div class="gpu-metric">
                        <span class="metric-label">MEM</span>
                        <div class="progress-bar"><div class="progress-fill {mem_class}" style="width: {mem_pct:.0f}%"></div></div>
                        <span class="metric-value">{mem_used:,}MB</span>
                    </div>
                    <div class="gpu-metric">
                        <span class="metric-label">TEMP</span>
                        <div class="progress-bar"><div class="progress-fill {temp_class}" style="width: {min(temp, 100)}%"></div></div>
                        <span class="metric-value">{temp}°C</span>
                    </div>
                    <div class="gpu-metric">
                        <span class="metric-label">PWR</span>
                        <div class="progress-bar"><div class="progress-fill accent" style="width: {power_pct:.0f}%"></div></div>
                        <span class="metric-value">{power:.0f}W</span>
                    </div>
                </div>
            </div>'''
        
        return f'''
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">🖥️</span> GPU Status</div>
            </div>
            <div class="panel-body">
                {gpus_html}
            </div>
        </div>'''
    
    def _build_breakdown_html(self, breakdown: Dict) -> str:
        """Build reward breakdown panel HTML."""
        if not breakdown.get("components"):
            return ""
        
        # Calculate max total for bar scaling
        max_abs_total = max(abs(c["total"]) for c in breakdown["components"]) if breakdown["components"] else 1
        if max_abs_total == 0:
            max_abs_total = 1
        
        cards_html = ""
        for comp in breakdown["components"]:
            bar_width = min(abs(comp["total"]) / max_abs_total * 100, 100)
            
            # Prepare description (escape for HTML)
            description = comp.get("description", "No description available.")
            description_html = description.replace('"', '&quot;').replace("'", "&#39;")
            card_id = f"reward-{comp['name']}"
            
            cards_html += f'''
            <div class="reward-card">
                <div class="reward-header">
                    <span class="reward-icon">{comp["icon"]}</span>
                    <span class="reward-name">{comp["display_name"]}</span>
                    <button class="reward-info-btn" onclick="toggleRewardInfo('{card_id}')" title="Show explanation">ℹ️</button>
                    <span class="reward-category {comp["category"]}">{comp["category"]}</span>
                </div>
                <div id="{card_id}" class="reward-description">
                    {description_html}
                </div>
                <div class="reward-stats">
                    <div class="reward-stat">
                        <div class="reward-stat-label">Total</div>
                        <div class="reward-stat-value" style="color: {comp['color']}">{comp["total"]:+.1f}</div>
                    </div>
                    <div class="reward-stat">
                        <div class="reward-stat-label">Avg/Step</div>
                        <div class="reward-stat-value">{comp["average"]:+.4f}</div>
                    </div>
                    <div class="reward-stat">
                        <div class="reward-stat-label">Active</div>
                        <div class="reward-stat-value">{comp["active_pct"]:.0f}%</div>
                    </div>
                </div>
                <div class="reward-bar">
                    <div class="reward-bar-fill" style="width: {bar_width}%; background: {comp['color']}"></div>
                </div>
            </div>'''
        
        return f'''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">🎯</span> Reward Components</div>
            </div>
            <div class="panel-body">
                <div class="reward-grid">
                    {cards_html}
                </div>
            </div>
        </div>'''
    
    def _build_snapshots_html(self) -> str:
        """Build snapshots gallery HTML."""
        if not self.stored_snapshots:
            return ""
        
        # Get latest snapshot
        latest = self.stored_snapshots[-1] if self.stored_snapshots else None
        if not latest or not latest.get("frames"):
            return ""
        
        frames_html = ""
        for frame in latest.get("frames", [])[:6]:
            step = frame.get("step", 0)
            pct = frame.get("pct", int(step / 12))  # Estimate if not provided
            reward = frame.get("reward", 0)
            distance = frame.get("distance", 0)
            detected = frame.get("detected", 0)
            collisions = frame.get("collisions", 0)
            
            frames_html += f'''
            <div class="snapshot-card">
                <img src="data:image/png;base64,{frame.get("image", "")}" alt="Step {step}">
                <div class="snapshot-info">
                    <div class="snapshot-header">{pct}% • Step {step}</div>
                    <div class="snapshot-metrics">
                        <span>R: {reward:.1f}</span>
                        <span>📏 {distance:.0f}m</span>
                        <span>💥 {collisions}</span>
                    </div>
                </div>
            </div>'''
        
        return f'''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">🎬</span> Latest Evaluation Snapshots</div>
                <span style="color: var(--text-dim); font-size: 0.8rem;">@ {latest.get("timestep", 0):,} steps</span>
            </div>
            <div class="panel-body">
                <div class="snapshot-grid">
                    {frames_html}
                </div>
            </div>
        </div>'''
    
    def _build_episode_metrics_html(self) -> str:
        """Build episode metrics panel HTML showing KPIs."""
        if not self.episode_metrics_history:
            return ""
        
        # Calculate aggregate stats from recent episodes
        recent = self.episode_metrics_history[-20:]  # Last 20 episodes
        
        win_rate = (self.wins / self.total_episodes * 100) if self.total_episodes > 0 else 0
        
        # Aggregate metrics
        avg_distance = sum(m.get("distance_total", 0) for m in recent) / len(recent)
        avg_detected_pct = sum(m.get("detected_time_pct", 0) for m in recent) / len(recent)
        avg_capture_progress = sum(m.get("final_capture_progress", 0) for m in recent) / len(recent)
        total_collisions = sum(m.get("collisions_total", 0) for m in recent)
        avg_tag_hit_rate = sum(m.get("tag_hit_rate_attacker", 0) for m in recent) / len(recent)
        avg_zone_time = sum(m.get("time_in_objective_zone", 0) for m in recent) / len(recent)
        
        # Count termination reasons
        reasons = {}
        for m in recent:
            reason = m.get("terminated_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        
        reason_badges = ""
        for reason, count in reasons.items():
            if reason == "captured":
                badge_class = "badge-success"
                icon = "🏆"
            elif reason == "timeout":
                badge_class = "badge-warning"
                icon = "⏱️"
            else:
                badge_class = "badge-danger"
                icon = "💔"
            reason_badges += f'<span class="badge {badge_class}">{icon} {reason}: {count}</span> '
        
        return f'''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📊</span> Episode Metrics (Last {len(recent)})</div>
            </div>
            <div class="panel-body">
                <div class="metrics-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                    <div class="metric-card" style="background: var(--bg-card); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--success);">
                        <div style="color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;">Win Rate</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {'var(--success)' if win_rate > 50 else 'var(--warning)'};">{win_rate:.0f}%</div>
                        <div style="color: var(--text-dim); font-size: 0.7rem;">{self.wins}/{self.total_episodes} episodes</div>
                    </div>
                    <div class="metric-card" style="background: var(--bg-card); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--accent-teal);">
                        <div style="color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;">Avg Capture Progress</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-teal);">{avg_capture_progress:.1f}s</div>
                    </div>
                    <div class="metric-card" style="background: var(--bg-card); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--accent-cyan);">
                        <div style="color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;">Avg Zone Time</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-cyan);">{avg_zone_time:.1f}s</div>
                    </div>
                    <div class="metric-card" style="background: var(--bg-card); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--accent-purple);">
                        <div style="color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;">Avg Distance</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-purple);">{avg_distance:.0f}m</div>
                    </div>
                    <div class="metric-card" style="background: var(--bg-card); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--warning);">
                        <div style="color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;">Detected Time</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {'var(--danger)' if avg_detected_pct > 50 else 'var(--success)'};">{avg_detected_pct:.0f}%</div>
                    </div>
                    <div class="metric-card" style="background: var(--bg-card); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--accent-orange);">
                        <div style="color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;">Tag Hit Rate</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-orange);">{avg_tag_hit_rate*100:.0f}%</div>
                    </div>
                    <div class="metric-card" style="background: var(--bg-card); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--danger);">
                        <div style="color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;">Collisions</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {'var(--danger)' if total_collisions > 10 else 'var(--text-primary)'};">{total_collisions}</div>
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <span style="color: var(--text-dim); font-size: 0.75rem; margin-right: 0.5rem;">Outcomes:</span>
                    {reason_badges}
                </div>
            </div>
        </div>'''
    
    def _build_recent_episodes_html(self) -> str:
        """Build recent episodes table HTML with full metrics."""
        if not self.episode_rewards:
            return ""
        
        # Get last 10 episodes with their metrics
        # All three lists are now kept in sync, so we can use the same indices
        n_recent = min(10, len(self.episode_rewards))
        
        rows_html = ""
        for i in range(n_recent - 1, -1, -1):  # Reverse order (newest first)
            # Use negative indexing to always get the most recent episodes
            list_idx = -(n_recent - i)
            reward = self.episode_rewards[list_idx]
            length = self.episode_lengths[list_idx]
            
            # Get episode metrics - lists are now synced, so same index works
            metrics = self.episode_metrics_history[list_idx] if self.episode_metrics_history else {}
            
            # Calculate the actual episode number for display
            ep_num = self.episodes_completed + list_idx + 1
            
            # Status badge based on actual win/termination
            win = metrics.get("win", False)
            reason = metrics.get("terminated_reason", "unknown")
            
            if win:
                badge = '<span class="badge badge-success">🏆 WIN</span>'
            elif reason == "timeout":
                badge = '<span class="badge badge-warning">⏱️ TIMEOUT</span>'
            elif reason == "all_disabled":
                badge = '<span class="badge badge-danger">💔 DISABLED</span>'
            elif reward > 0:
                badge = '<span class="badge badge-warning">⚡ PARTIAL</span>'
            else:
                badge = '<span class="badge badge-danger">❌ LOSS</span>'
            
            # Extract key metrics
            distance = metrics.get("distance_total", 0)
            zone_time = metrics.get("time_in_objective_zone", 0)
            detected_pct = metrics.get("detected_time_pct", 0)
            collisions = metrics.get("collisions_total", 0)
            capture_prog = metrics.get("final_capture_progress", 0)
            tag_rate = metrics.get("tag_hit_rate_attacker", 0) * 100
            
            # Color-code detected time
            detected_color = "var(--success)" if detected_pct < 20 else ("var(--warning)" if detected_pct < 50 else "var(--danger)")
            collision_color = "var(--success)" if collisions == 0 else ("var(--warning)" if collisions < 50 else "var(--danger)")
            
            rows_html += f'''
            <tr>
                <td>#{ep_num}</td>
                <td style="font-family: 'JetBrains Mono', monospace; color: {'var(--success)' if reward > 50 else ('var(--warning)' if reward > 0 else 'var(--danger)')};">{reward:.1f}</td>
                <td>{length}</td>
                <td style="font-family: 'JetBrains Mono', monospace;">{distance:.0f}m</td>
                <td style="font-family: 'JetBrains Mono', monospace;">{zone_time:.1f}s</td>
                <td style="font-family: 'JetBrains Mono', monospace; color: {detected_color};">{detected_pct:.0f}%</td>
                <td style="font-family: 'JetBrains Mono', monospace; color: {collision_color};">{collisions}</td>
                <td style="font-family: 'JetBrains Mono', monospace;">{capture_prog:.0f}%</td>
                <td>{badge}</td>
            </tr>'''
        
        return f'''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📊</span> Recent Episodes</div>
            </div>
            <div class="panel-body" style="padding: 0; overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Episode</th>
                            <th>Reward</th>
                            <th>Steps</th>
                            <th>Distance</th>
                            <th>Zone Time</th>
                            <th>Detected</th>
                            <th>Collisions</th>
                            <th>Capture</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>'''
    
    def _build_commands_html(self, run_name: str) -> str:
        """Build quick commands panel with useful scripts."""
        run_path = f"runs/{run_name}"
        model_path = f"{run_path}/final_model"
        checkpoint_path = f"{run_path}/checkpoints"
        logs_path = f"{run_path}/logs"
        
        # Try to find latest checkpoint
        latest_checkpoint = None
        latest_checkpoint_name = None
        latest_checkpoint_model_path = None
        if self.run_dir:
            checkpoint_dir = self.run_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("ppo_mission_*.zip"))
                if checkpoints:
                    # Sort by modification time, get latest
                    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    latest_checkpoint_name = latest_checkpoint.stem  # Remove .zip
                    # Model path is without .zip extension
                    latest_checkpoint_model_path = str(latest_checkpoint.with_suffix(''))
                    latest_checkpoint = str(latest_checkpoint)
        
        commands = []
        
        # Add latest checkpoint evaluation if available
        if latest_checkpoint_model_path:
            # Extract step count for display
            steps_str = latest_checkpoint_name.split('_')[-2] if latest_checkpoint_name else "unknown"
            commands.append({
                "name": "⚡ Eval Latest Checkpoint",
                "desc": f"Evaluate current checkpoint ({latest_checkpoint_name})",
                "cmd": f"python -m mission_gym.scripts.evaluate --model {latest_checkpoint_model_path} --episodes 10",
            })
            commands.append({
                "name": "🎬 Video Latest Checkpoint",
                "desc": f"Record video of latest checkpoint",
                "cmd": f"python -m mission_gym.scripts.record_video --model {latest_checkpoint_model_path} --episodes 3",
            })
        
        # Add final model commands (for completed training)
        commands.extend([
            {
                "name": "🎮 Eval Final Model",
                "desc": "Evaluate final model (after training completes)",
                "cmd": f"python -m mission_gym.scripts.evaluate --model {model_path} --episodes 10",
            },
            {
                "name": "🎬 Video Final Model",
                "desc": "Record video of final model (after training completes)",
                "cmd": f"python -m mission_gym.scripts.record_video --model {model_path} --episodes 3",
            },
        ])
        
        # Add other utility commands
        commands.extend([
            {
                "name": "🕹️ Play Manual",
                "desc": "Control units manually with keyboard",
                "cmd": "python -m mission_gym.scripts.play_manual",
            },
            {
                "name": "📊 TensorBoard",
                "desc": "View training graphs in browser",
                "cmd": f"tensorboard --logdir {logs_path}",
            },
            {
                "name": "📋 List Checkpoints",
                "desc": "Show saved model checkpoints",
                "cmd": f"ls -lht {checkpoint_path} | head -20",
            },
        ])
        
        # Add resume command if we have a checkpoint and can read metadata
        if latest_checkpoint_model_path and self.run_dir:
            metadata_file = self.run_dir / "run_metadata.json"
            resume_cmd = f"python -m mission_gym.scripts.train_ppo --timesteps 500000 --load-checkpoint {latest_checkpoint_model_path}"
            
            # Try to read training config to provide accurate resume command
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    n_envs = metadata.get('n_envs', 64)
                    network_arch = metadata.get('network_arch', '1024,512,512,256')
                    n_epochs = metadata.get('n_epochs', 30)
                    
                    resume_cmd = f"python -m mission_gym.scripts.train_ppo --timesteps 50000000 --n-envs {n_envs} --subproc --network-arch '{network_arch}' --n-epochs {n_epochs} --load-checkpoint {latest_checkpoint_model_path} --notes 'Resumed from {latest_checkpoint_name}'"
                except:
                    pass
            
            commands.append({
                "name": "🔄 Resume Training",
                "desc": f"Resume from latest checkpoint ({latest_checkpoint_name})",
                "cmd": resume_cmd,
            })
        
        commands.append({
            "name": "🌿 Branch Training",
            "desc": "Start new branch from latest checkpoint",
            "cmd": f"python -m mission_gym.scripts.train_ppo --timesteps 50000000 --n-envs 32 --subproc --parent-checkpoint {latest_checkpoint_model_path if latest_checkpoint_model_path else checkpoint_path + '/ppo_mission_XXXXX_steps'} --branch-name {run_name}-experiment --notes 'New experiment branch'",
        })
        
        commands_html = ""
        for cmd in commands:
            commands_html += f'''
            <div class="command-card">
                <div class="command-header">
                    <span class="command-name">{cmd["name"]}</span>
                    <button class="copy-btn" onclick="copyCommand(this)" data-cmd="{cmd["cmd"]}">📋 Copy</button>
                </div>
                <div class="command-desc">{cmd["desc"]}</div>
                <code class="command-code">{cmd["cmd"]}</code>
            </div>'''
        
        return f'''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">⚡</span> Quick Commands</div>
            </div>
            <div class="panel-body">
                <div class="commands-grid">
                    {commands_html}
                </div>
            </div>
        </div>
        <style>
            .commands-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 1rem;
            }}
            .command-card {{
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 1rem;
            }}
            .command-card:hover {{
                border-color: var(--accent-teal);
            }}
            .command-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }}
            .command-name {{
                font-weight: 600;
                color: var(--accent-teal);
            }}
            .copy-btn {{
                background: var(--bg-panel);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 0.25rem 0.5rem;
                color: var(--text-secondary);
                cursor: pointer;
                font-size: 0.75rem;
            }}
            .copy-btn:hover {{
                background: var(--accent-teal);
                color: var(--bg-dark);
                border-color: var(--accent-teal);
            }}
            .copy-btn.copied {{
                background: var(--success);
                border-color: var(--success);
                color: white;
            }}
            .command-desc {{
                color: var(--text-secondary);
                font-size: 0.8rem;
                margin-bottom: 0.5rem;
            }}
            .command-code {{
                display: block;
                background: var(--bg-dark);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 0.5rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                color: var(--accent-cyan);
                overflow-x: auto;
                white-space: nowrap;
            }}
        </style>
        <script>
            function copyCommand(btn) {{
                const cmd = btn.getAttribute('data-cmd');
                navigator.clipboard.writeText(cmd).then(() => {{
                    btn.textContent = '✓ Copied!';
                    btn.classList.add('copied');
                    setTimeout(() => {{
                        btn.textContent = '📋 Copy';
                        btn.classList.remove('copied');
                    }}, 2000);
                }});
            }}
        </script>'''
    
    def _build_action_log_html(self) -> str:
        """Build action log panel showing last 100 commands."""
        if not self.action_log:
            return '''
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📜</span> Action Log (Last 100)</div>
            </div>
            <div class="panel-body">
                <p style="color: var(--text-secondary);">No actions recorded yet. Training will populate this log.</p>
            </div>
        </div>'''
        
        # Action name mapping (index to name for display)
        action_names = {
            0: "NOOP", 1: "THROT↑", 2: "THROT↓", 3: "LEFT", 4: "RIGHT", 
            5: "BRAKE", 6: "HOLD", 7: "ALT↑", 8: "ALT↓"
        }
        
        # Build table rows (most recent first)
        rows_html = ""
        for entry in reversed(self.action_log[-50:]):  # Show last 50 in table
            timestep = entry["timestep"]
            actions = entry["actions"]
            rewards = entry["rewards"]
            dones = entry.get("dones", [])
            
            # Format actions as colored badges per unit
            # Handle both flat actions [a1, a2, a3, a4] and nested [[a1,a2,a3,a4], ...]
            action_badges = ""
            flat_actions = actions
            
            # If actions is nested (from vectorized env), flatten first row
            if actions and isinstance(actions[0], (list, tuple)):
                flat_actions = actions[0] if len(actions) > 0 else []
            
            for i, action in enumerate(flat_actions):
                # Handle if action is still a list/array
                if isinstance(action, (list, tuple)):
                    action = action[0] if action else 0
                action = int(action)  # Ensure it's an int
                
                action_name = action_names.get(action, f"A{action}")
                # Color code by action type
                if action == 0:  # NOOP
                    color = "var(--text-dim)"
                elif action in [1, 2]:  # Throttle
                    color = "var(--accent-cyan)"
                elif action in [3, 4]:  # Turn
                    color = "var(--accent-teal)"
                elif action in [5, 6]:  # Brake/Hold
                    color = "var(--accent-purple)"
                else:  # Alt
                    color = "var(--accent-orange)"
                action_badges += f'<span class="action-badge" style="background: {color}20; color: {color}; border: 1px solid {color}40;">{action_name}</span>'
            
            # Format rewards - handle nested lists from vectorized envs
            try:
                if rewards and isinstance(rewards[0], (list, tuple)):
                    total_reward = float(rewards[0][0]) if rewards[0] else 0
                else:
                    total_reward = sum(float(r) for r in rewards) if rewards else 0
            except (TypeError, IndexError):
                total_reward = 0
            reward_color = "var(--success)" if total_reward > 0 else ("var(--danger)" if total_reward < 0 else "var(--text-dim)")
            
            # Check for episode end - handle nested lists
            done_marker = ""
            try:
                if dones:
                    # Flatten if nested
                    flat_dones = dones[0] if isinstance(dones[0], (list, tuple)) else dones
                    if any(flat_dones):
                        done_marker = '<span style="color: var(--warning); margin-left: 0.5rem;">🏁</span>'
            except (TypeError, IndexError):
                pass
            
            rows_html += f'''
                <tr>
                    <td style="font-family: 'JetBrains Mono', monospace; color: var(--text-secondary);">{timestep:,}</td>
                    <td><div class="action-group">{action_badges}</div></td>
                    <td style="color: {reward_color}; font-family: 'JetBrains Mono', monospace;">{total_reward:+.3f}{done_marker}</td>
                </tr>'''
        
        return f'''
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📜</span> Action Log (Last 50 Steps)</div>
                <span class="badge" style="background: var(--bg-card);">{len(self.action_log)} recorded</span>
            </div>
            <div class="panel-body" style="max-height: 400px; overflow-y: auto;">
                <table style="font-size: 0.8rem;">
                    <thead>
                        <tr>
                            <th style="width: 100px;">Step</th>
                            <th>Unit Actions</th>
                            <th style="width: 100px;">Reward</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        <style>
            .action-group {{
                display: flex;
                gap: 0.25rem;
                flex-wrap: wrap;
            }}
            .action-badge {{
                padding: 0.15rem 0.4rem;
                border-radius: 4px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                font-weight: 500;
            }}
        </style>'''
    
    def _build_continuation_info_html(self) -> str:
        """Build policy continuation information panel."""
        return '''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">🌿</span> Policy Continuation Guide</div>
            </div>
            <div class="panel-body">
                <div style="font-size: 0.9rem; line-height: 1.6;">
                    <p style="margin-bottom: 1rem; color: var(--text-secondary);">
                        Branch from this policy to continue training with modified configurations. 
                        <a href="../docs/API_CONTINUATION_RULES.md" target="_blank" style="color: var(--primary); text-decoration: none;">📖 Full Documentation</a>
                    </p>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem;">
                        <div>
                            <div style="color: var(--success); font-weight: 600; margin-bottom: 0.5rem;">✅ Safe Changes (Won't Break Compatibility)</div>
                            <ul style="margin: 0; padding-left: 1.5rem; color: var(--text-secondary); font-size: 0.85rem;">
                                <li>Reward weights (<code>reward.yaml</code>)</li>
                                <li>Enable/disable flags (<code>engagement.yaml</code>)</li>
                                <li>Termination config (stagnation time)</li>
                                <li>Physics parameters (speeds, turn rates)</li>
                                <li>Sensor configs (ranges, FOV)</li>
                                <li>Obstacle positions</li>
                            </ul>
                        </div>
                        <div>
                            <div style="color: var(--error); font-weight: 600; margin-bottom: 0.5rem;">❌ Breaking Changes (Will Fail)</div>
                            <ul style="margin: 0; padding-left: 1.5rem; color: var(--text-secondary); font-size: 0.85rem;">
                                <li>Number of units (changes obs/action space)</li>
                                <li>Adding/removing actions from unit configs</li>
                                <li>Changing observation features</li>
                            </ul>
                            <p style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--warning);">
                                💡 <strong>Tip:</strong> Use <code>enable.tag: false</code> instead of removing TAG action!
                            </p>
                        </div>
                    </div>
                    
                    <div style="background: var(--bg-secondary); padding: 1rem; border-radius: 6px; border-left: 3px solid var(--primary);">
                        <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--primary);">📝 Example: Branch from this policy</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; background: var(--bg); padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem; overflow-x: auto;">
                            <div style="color: #6A9955;"># 1. Modify config (e.g., increase zone bonus)</div>
                            <div>vim configs/reward.yaml</div>
                            <div style="margin-top: 0.5rem; color: #6A9955;"># 2. Branch from this policy</div>
                            <div>python -m mission_gym.scripts.train_ppo \\</div>
                            <div style="padding-left: 1rem;">--parent-checkpoint <span style="color: var(--warning);">[see "Resume Training" above]</span> \\</div>
                            <div style="padding-left: 1rem;">--branch-name higher-zone-bonus \\</div>
                            <div style="padding-left: 1rem;">--timesteps 500000 \\</div>
                            <div style="padding-left: 1rem;">--notes "Increased zone_time to 5.0"</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 1rem; padding: 0.75rem; background: #2d3748; border-radius: 6px; border: 1px solid #4a5568;">
                        <div style="font-size: 0.85rem; color: #e2e8f0;">
                            <strong>📊 Episode Outcomes:</strong> This run tracks 5 termination types:
                            <code style="margin-left: 0.5rem; padding: 0.15rem 0.4rem; background: var(--bg); border-radius: 3px;">captured</code>
                            <code style="margin-left: 0.25rem; padding: 0.15rem 0.4rem; background: var(--bg); border-radius: 3px;">early_success</code>
                            <code style="margin-left: 0.25rem; padding: 0.15rem 0.4rem; background: var(--bg); border-radius: 3px;">stalled</code>
                            <code style="margin-left: 0.25rem; padding: 0.15rem 0.4rem; background: var(--bg); border-radius: 3px;">all_disabled</code>
                            <code style="margin-left: 0.25rem; padding: 0.15rem 0.4rem; background: var(--bg); border-radius: 3px;">timeout</code>
                        </div>
                    </div>
                </div>
            </div>
        </div>'''
    
    def _build_lineage_html(self) -> str:
        """Build policy lineage tree panel."""
        if not self.run_dir:
            return ""
        
        # Import build_lineage_tree_html from run_utils
        from mission_gym.scripts.run_utils import build_lineage_tree_html, get_runs_dir
        
        try:
            # Collect all runs
            runs_dir = get_runs_dir()
            runs_data = []
            
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                lineage_file = run_dir / "lineage.json"
                metadata_file = run_dir / "run_metadata.json"
                
                if not lineage_file.exists() and not metadata_file.exists():
                    continue
                
                run_info = {
                    "name": run_dir.name,
                    "parent": None,
                    "created": None,
                    "timesteps": 0,
                    "lineage": {},
                }
                
                if lineage_file.exists():
                    try:
                        with open(lineage_file) as f:
                            lineage = json.load(f)
                            run_info["parent"] = lineage.get("parent_run_name")
                            run_info["created"] = lineage.get("created_at")
                            run_info["lineage"] = lineage
                    except Exception:
                        pass
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            run_info["timesteps"] = metadata.get("args", {}).get("timesteps", 0)
                            if not run_info["created"]:
                                run_info["created"] = metadata.get("created_at")
                    except Exception:
                        pass
                
                runs_data.append(run_info)
            
            # Generate lineage tree HTML for current run
            lineage_tree_html = build_lineage_tree_html(runs_data, self.run_dir.name)
            
            return f'''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">🌳</span> Policy Lineage Tree</div>
            </div>
            <div class="panel-body">
                <div style="font-size: 0.85rem; line-height: 1.8; font-family: 'JetBrains Mono', monospace;">
                    {lineage_tree_html}
                </div>
            </div>
        </div>'''
        except Exception as e:
            return f'''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">🌳</span> Policy Lineage Tree</div>
            </div>
            <div class="panel-body">
                <p style="color: var(--text-secondary);">Unable to load lineage tree: {str(e)}</p>
            </div>
        </div>'''
    
    def _build_config_html(self) -> str:
        """Build configuration viewer HTML."""
        if not self.config_yaml:
            return ""
        
        tabs_html = ""
        panels_html = ""
        
        for i, (filename, content) in enumerate(self.config_yaml.items()):
            file_id = filename.replace('.', '_').replace('/', '_')
            active = "active" if i == 0 else ""
            display = "block" if i == 0 else "none"
            
            short_name = filename.replace('.yaml', '').replace('_', ' ').title()
            
            tabs_html += f'<button class="config-tab {active}" onclick="showConfig(\'{file_id}\')">{short_name}</button>'
            
            escaped = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            panels_html += f'<pre class="config-content" id="{file_id}" style="display: {display};">{escaped}</pre>'
        
        return f'''
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">⚙️</span> Configuration</div>
            </div>
            <div class="panel-body">
                <div class="config-tabs">
                    {tabs_html}
                </div>
                {panels_html}
            </div>
        </div>'''


class EvalWithMonitorCallback(BaseCallback):
    """
    Evaluation callback that also updates the HTML monitor.
    Captures simulation snapshots during evaluation.
    """
    
    def __init__(
        self,
        eval_env,
        html_monitor: HTMLMonitorCallback,
        n_eval_episodes: int = 5,
        eval_freq: int = 5000,
        capture_snapshots: bool = True,
        snapshot_steps: List[int] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.html_monitor = html_monitor
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.last_eval_timestep = 0
        self.capture_snapshots = capture_snapshots and PYGAME_AVAILABLE
        # Capture snapshots at key episode percentages (1200 steps = 5 min episode)
        # 10%, 30%, 50%, 75%, 100% of episode = steps 120, 360, 600, 900, 1199
        self.snapshot_steps = snapshot_steps or [120, 360, 600, 900, 1199]
        
        self.snapshots: List[Dict] = []
        self.max_snapshots = 8
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self._evaluate()
            self.last_eval_timestep = self.num_timesteps
        return True
    
    def _evaluate(self) -> None:
        """Run evaluation episodes with snapshot capture."""
        episode_rewards = []
        captured_frames = []
        
        for ep_idx in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            step = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                if ep_idx == 0 and self.capture_snapshots and step in self.snapshot_steps:
                    # Calculate episode percentage
                    max_steps = 1200  # Default episode length
                    try:
                        max_steps = self.eval_env.max_steps
                    except:
                        pass
                    pct = int(step / max_steps * 100)
                    
                    # Get metrics from info
                    metrics = info.get("episode_metrics", {})
                    
                    frame = self._capture_frame(step, total_reward, info, pct, metrics)
                    if frame:
                        captured_frames.append(frame)
                
                step += 1
            
            episode_rewards.append(total_reward)
        
        mean_reward = np.mean(episode_rewards)
        
        if self.verbose > 0:
            print(f"Eval at {self.num_timesteps}: mean_reward={mean_reward:.2f}")
        
        if captured_frames:
            self.snapshots.append({
                "timestep": self.num_timesteps,
                "reward": mean_reward,
                "frames": captured_frames,
            })
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
        
        self.html_monitor.add_eval_result(mean_reward, self.num_timesteps, self.snapshots)
    
    def _capture_frame(self, step: int, reward: float, info: dict, pct: int = 0, metrics: dict = None) -> Optional[Dict]:
        """Capture a frame from the environment as base64."""
        if not PYGAME_AVAILABLE:
            return None
        
        try:
            import pygame
            
            if not pygame.get_init():
                import os
                if 'SDL_VIDEODRIVER' not in os.environ:
                    os.environ['SDL_VIDEODRIVER'] = 'dummy'
                pygame.init()
            if not pygame.font.get_init():
                pygame.font.init()
            
            env = self.eval_env
            config = env.config
            size = 400
            
            screen = pygame.Surface((size, size))
            self._render_to_surface(screen, env, size, step, reward)
            
            png_bytes = io.BytesIO()
            pygame.image.save(screen, png_bytes, "PNG")
            png_bytes.seek(0)
            
            b64 = base64.b64encode(png_bytes.read()).decode('utf-8')
            
            # Extract key metrics for display
            distance = metrics.get("distance_total", 0) if metrics else 0
            detected = metrics.get("detected_time_pct", 0) if metrics else 0
            collisions = metrics.get("collisions_total", 0) if metrics else 0
            
            return {
                "step": step,
                "pct": pct,
                "reward": reward,
                "image": b64,
                "distance": distance,
                "detected": detected,
                "collisions": collisions,
            }
        except Exception as e:
            if self.verbose > 0:
                print(f"Failed to capture frame: {e}")
            return None
    
    def _render_to_surface(self, screen, env, size: int, step: int, reward: float):
        """Render environment state to pygame surface."""
        import pygame
        import math
        
        BG = (10, 14, 20)
        GRID = (42, 53, 68)
        OBSTACLE = (80, 90, 110)
        OBJECTIVE = (0, 100, 80)
        OBJECTIVE_CAPTURED = (0, 212, 170)
        ATTACKER_UGV = (0, 191, 255)
        ATTACKER_UAV = (100, 200, 255)
        DEFENDER = (239, 68, 68)
        DISABLED = (72, 79, 88)
        TEXT = (230, 237, 243)
        
        config = env.config
        world_w = config.world.width
        world_h = config.world.height
        scale = size / max(world_w, world_h)
        
        def w2s(x, y):
            return int(x * scale), int((world_h - y) * scale)
        
        screen.fill(BG)
        
        for x in range(0, int(world_w) + 1, 20):
            pygame.draw.line(screen, GRID, w2s(x, 0), w2s(x, world_h), 1)
        for y in range(0, int(world_h) + 1, 20):
            pygame.draw.line(screen, GRID, w2s(0, y), w2s(world_w, y), 1)
        
        for obs in config.world.obstacles:
            if obs.type == "circle":
                pygame.draw.circle(screen, OBSTACLE, w2s(obs.x, obs.y), int(obs.radius * scale))
            elif obs.type == "rect":
                corners = obs.get_corners()
                if corners:
                    pygame.draw.polygon(screen, OBSTACLE, [w2s(x, y) for x, y in corners])
        
        obj = env.objective
        progress = obj.capture_progress / obj.capture_time_required
        color = tuple(int(OBJECTIVE[i] + (OBJECTIVE_CAPTURED[i] - OBJECTIVE[i]) * progress) for i in range(3))
        pygame.draw.circle(screen, color, w2s(obj.x, obj.y), int(obj.radius * scale))
        pygame.draw.circle(screen, TEXT, w2s(obj.x, obj.y), int(obj.radius * scale), 2)
        
        for unit in env.defenders + env.attackers:
            ux, uy = w2s(unit.x, unit.y)
            
            if unit.is_disabled:
                color = DISABLED
            elif unit.team == "attacker":
                color = ATTACKER_UAV if unit.category == "air" else ATTACKER_UGV
            else:
                color = DEFENDER
            
            radius = max(int((unit.type_config.radius if unit.type_config else 1.0) * scale * 1.5), 4)
            
            if unit.category == "air" and unit.altitude > 0:
                pygame.draw.circle(screen, (20, 20, 30), (ux + unit.altitude * 2, uy + unit.altitude * 2), radius)
            
            pygame.draw.circle(screen, color, (ux, uy), radius)
            
            heading_rad = math.radians(unit.heading)
            dx = math.cos(heading_rad) * radius * 1.5
            dy = -math.sin(heading_rad) * radius * 1.5
            pygame.draw.line(screen, TEXT, (ux, uy), (int(ux + dx), int(uy + dy)), 2)
        
        font = pygame.font.Font(None, 20)
        screen.blit(font.render(f"Step: {step}", True, TEXT), (5, 5))
        screen.blit(font.render(f"R: {reward:.1f}", True, TEXT), (5, 22))


class MetricsCallback(BaseCallback):
    """
    Callback that logs episode metrics to TensorBoard and prints beautiful console output.
    
    Uses Rich library for beautiful formatted tables and panels.
    
    Logs key performance indicators from the EpisodeMetrics collected by the env:
    - Mission outcomes (win rate, time to capture)
    - Engagement stats (tag hit rates)
    - Detection/stealth metrics
    - Fleet performance (distance, collisions)
    """
    
    def __init__(self, verbose: int = 0, print_freq: int = 10):
        super().__init__(verbose)
        self.episode_count = 0
        self.wins = 0
        self.print_freq = print_freq
        self.recent_metrics: List[Dict] = []
        self.last_print_episode = 0
        self.console = Console() if RICH_AVAILABLE else None
    
    def _print_metrics_summary(self) -> None:
        """Print a beautiful summary of recent metrics using Rich."""
        if not self.recent_metrics:
            return
        
        recent = self.recent_metrics[-self.print_freq:]
        
        # Calculate aggregates
        wins = sum(1 for m in recent if m.get("win", False))
        win_rate = wins / len(recent) * 100
        overall_win_rate = self.wins / self.episode_count * 100 if self.episode_count > 0 else 0
        avg_capture = sum(m.get("final_capture_progress", 0) for m in recent) / len(recent)
        avg_zone_time = sum(m.get("time_in_objective_zone", 0) for m in recent) / len(recent)
        avg_distance = sum(m.get("distance_total", 0) for m in recent) / len(recent)
        avg_detected = sum(m.get("detected_time_pct", 0) for m in recent) / len(recent)
        total_collisions = sum(m.get("collisions_total", 0) for m in recent)
        avg_tag_rate = sum(m.get("tag_hit_rate_attacker", 0) for m in recent) / len(recent) * 100
        avg_steps = sum(m.get("episode_steps", 0) for m in recent) / len(recent)
        
        # Termination reasons
        reasons = {"captured": 0, "timeout": 0, "all_disabled": 0}
        for m in recent:
            r = m.get("terminated_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        
        if RICH_AVAILABLE and self.console:
            self._print_rich_summary(
                recent, wins, win_rate, overall_win_rate, avg_capture, avg_zone_time,
                avg_distance, avg_detected, total_collisions, avg_tag_rate, avg_steps, reasons
            )
        else:
            self._print_simple_summary(
                recent, wins, win_rate, avg_capture, avg_distance, avg_detected, 
                total_collisions, reasons
            )
    
    def _print_rich_summary(
        self, recent, wins, win_rate, overall_win_rate, avg_capture, avg_zone_time,
        avg_distance, avg_detected, total_collisions, avg_tag_rate, avg_steps, reasons
    ) -> None:
        """Print beautiful Rich-formatted summary."""
        # Win rate color
        if win_rate >= 50:
            wr_style = "bold green"
        elif win_rate >= 20:
            wr_style = "bold yellow"
        else:
            wr_style = "bold red"
        
        # Create main metrics table
        table = Table(
            title=f"[bold cyan]📊 Episode Metrics[/] [dim](#{self.episode_count - len(recent) + 1}-{self.episode_count})[/]",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold white",
            show_header=True,
            padding=(0, 1),
        )
        
        # KPI section
        table.add_column("🎯 KPI", style="bold", width=14)
        table.add_column("Value", justify="right", width=10)
        table.add_column("🚀 Fleet", style="bold", width=14)
        table.add_column("Value", justify="right", width=10)
        table.add_column("⚔️ Combat", style="bold", width=14)
        table.add_column("Value", justify="right", width=10)
        
        # Outcome string
        outcome_parts = []
        if reasons.get("captured", 0) > 0:
            outcome_parts.append(f"[green]🏆 {reasons['captured']}[/]")
        if reasons.get("timeout", 0) > 0:
            outcome_parts.append(f"[yellow]⏱️ {reasons['timeout']}[/]")
        if reasons.get("all_disabled", 0) > 0:
            outcome_parts.append(f"[red]💔 {reasons['all_disabled']}[/]")
        outcome_str = " ".join(outcome_parts) if outcome_parts else "[dim]none[/]"
        
        # Add rows
        table.add_row(
            "Win Rate", f"[{wr_style}]{win_rate:.0f}%[/]",
            "Distance", f"[cyan]{avg_distance:.0f}m[/]",
            "Tag Rate", f"[magenta]{avg_tag_rate:.0f}%[/]",
        )
        table.add_row(
            "Overall", f"[cyan]{self.wins}/{self.episode_count}[/]",
            "Collisions", f"[{'red' if total_collisions > 10 else 'green'}]{total_collisions}[/]",
            "Detected", f"[{'red' if avg_detected > 50 else 'green'}]{avg_detected:.0f}%[/]",
        )
        table.add_row(
            "Capture", f"[green]{avg_capture:.1f}s[/]",
            "Zone Time", f"[cyan]{avg_zone_time:.1f}s[/]",
            "Outcomes", outcome_str,
        )
        table.add_row(
            "Steps/Ep", f"[dim]{avg_steps:.0f}[/]",
            "", "",
            "", "",
        )
        
        self.console.print()
        self.console.print(table)
    
    def _print_simple_summary(
        self, recent, wins, win_rate, avg_capture, avg_distance, avg_detected, 
        total_collisions, reasons
    ) -> None:
        """Fallback simple summary without Rich."""
        print()
        print(f"  {'─' * 60}")
        print(f"  📊 METRICS (Episodes {self.episode_count - len(recent) + 1}-{self.episode_count})")
        print(f"  {'─' * 60}")
        print(f"    Win Rate: {win_rate:5.1f}% ({wins}/{len(recent)})  │  Overall: {self.wins}/{self.episode_count}")
        print(f"    Capture:  {avg_capture:5.1f}s   │  Distance: {avg_distance:6.0f}m  │  Detected: {avg_detected:4.0f}%")
        outcome_parts = []
        for r, count in reasons.items():
            if count > 0:
                outcome_parts.append(f"{r}:{count}")
        print(f"    Outcomes: {' '.join(outcome_parts)}  │  Collisions: {total_collisions}")
        print(f"  {'─' * 60}")
    
    def _on_step(self) -> bool:
        """Log metrics from completed episodes."""
        infos = self.locals.get("infos", [])
        
        for info in infos:
            metrics = info.get("episode_metrics")
            if not metrics:
                continue
            
            self.episode_count += 1
            if metrics.get("win", False):
                self.wins += 1
            
            # Store for console printing
            self.recent_metrics.append(metrics)
            if len(self.recent_metrics) > 100:
                self.recent_metrics = self.recent_metrics[-100:]
            
            # Print summary every N episodes
            if self.episode_count - self.last_print_episode >= self.print_freq:
                self._print_metrics_summary()
                self.last_print_episode = self.episode_count
            
            # Mission outcome KPIs
            self.logger.record("kpi/win", float(metrics.get("win", False)))
            self.logger.record("kpi/win_rate", self.wins / self.episode_count)
            self.logger.record("kpi/episode_steps", metrics.get("episode_steps", 0))
            self.logger.record("kpi/episode_sim_time", metrics.get("episode_sim_time", 0.0))
            
            # Objective metrics
            self.logger.record("kpi/final_capture_progress", metrics.get("final_capture_progress", 0.0))
            time_to_capture = metrics.get("time_to_capture")
            if time_to_capture is not None:
                self.logger.record("kpi/time_to_capture", time_to_capture)
            self.logger.record("kpi/time_in_objective_zone", metrics.get("time_in_objective_zone", 0.0))
            
            # Fleet status
            self.logger.record("fleet/attackers_alive_end", metrics.get("num_attackers_alive_end", 0))
            self.logger.record("fleet/attackers_disabled", metrics.get("num_attackers_disabled_total", 0))
            self.logger.record("fleet/distance_total", metrics.get("distance_total", 0.0))
            self.logger.record("fleet/distance_mean", metrics.get("distance_mean", 0.0))
            
            # Engagement stats
            self.logger.record("engagement/tag_attempts_attacker", metrics.get("tag_attempts_attacker", 0))
            self.logger.record("engagement/tag_hits_attacker", metrics.get("tag_hits_attacker", 0))
            self.logger.record("engagement/tag_hit_rate_attacker", metrics.get("tag_hit_rate_attacker", 0.0))
            self.logger.record("engagement/tag_attempts_defender", metrics.get("tag_attempts_defender", 0))
            self.logger.record("engagement/tag_hits_defender", metrics.get("tag_hits_defender", 0))
            self.logger.record("engagement/tag_hit_rate_defender", metrics.get("tag_hit_rate_defender", 0.0))
            self.logger.record("engagement/disable_events", metrics.get("disable_events", 0))
            
            # Detection / stealth
            self.logger.record("stealth/detected_time", metrics.get("detected_time", 0.0))
            self.logger.record("stealth/detected_time_pct", metrics.get("detected_time_pct", 0.0))
            self.logger.record("stealth/detection_events", metrics.get("detection_events", 0))
            first_detect = metrics.get("first_detect_time")
            if first_detect is not None:
                self.logger.record("stealth/first_detect_time", first_detect)
            
            # Collisions / violations
            self.logger.record("safety/collisions_total", metrics.get("collisions_total", 0))
            self.logger.record("safety/integrity_lost_total", metrics.get("integrity_lost_total", 0.0))
            
            # Termination reason distribution
            reason = metrics.get("terminated_reason", "unknown")
            self.logger.record(f"outcome/is_{reason}", 1.0)
        
        return True


class RichOutputFormat:
    """
    Custom SB3 output format using Rich for beautiful console logging.
    
    Replaces the default dashed-box format with modern Rich tables grouped by category.
    """
    
    # Category configuration: (title, style, icon)
    CATEGORIES = {
        "kpi": ("🎯 Key Performance", "bold cyan", "cyan"),
        "fleet": ("🚀 Fleet Status", "bold blue", "blue"),
        "engagement": ("⚔️  Engagement", "bold magenta", "magenta"),
        "stealth": ("👁️  Detection", "bold yellow", "yellow"),
        "safety": ("🛡️  Safety", "bold red", "red"),
        "outcome": ("📋 Outcome", "bold green", "green"),
        "train": ("🧠 Training", "bold white", "white"),
        "time": ("⏱️  Timing", "dim", "dim"),
        "rollout": ("📦 Rollout", "bold", "white"),
    }
    
    def __init__(self, suppress_categories: Optional[List[str]] = None):
        """
        Args:
            suppress_categories: List of categories to not print (e.g., ["time", "rollout"])
        """
        self.console = Console() if RICH_AVAILABLE else None
        self.suppress_categories = set(suppress_categories or [])
    
    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, bool], step: int) -> None:
        """Write logged values using Rich formatting."""
        if not RICH_AVAILABLE or not self.console:
            self._write_simple(key_values, step)
            return
        
        # Group metrics by category
        categories: Dict[str, Dict[str, Any]] = {}
        uncategorized: Dict[str, Any] = {}
        
        for key, value in sorted(key_values.items()):
            if key_excluded.get(key, False):
                continue
            
            # Skip step key
            if key == "step":
                continue
            
            # Parse category from key (e.g., "kpi/win_rate" -> "kpi", "win_rate")
            if "/" in key:
                cat, metric = key.split("/", 1)
            else:
                cat, metric = "", key
            
            # Skip suppressed categories
            if cat in self.suppress_categories:
                continue
            
            if cat in self.CATEGORIES:
                if cat not in categories:
                    categories[cat] = {}
                categories[cat][metric] = value
            else:
                uncategorized[key] = value
        
        # If nothing to print, skip
        if not categories and not uncategorized:
            return
        
        self.console.print()
        
        # Create a grid layout
        main_table = Table(
            title=f"[bold cyan]━━━ Training Step {step:,} ━━━[/]",
            box=box.SIMPLE,
            border_style="dim",
            show_header=False,
            padding=(0, 1),
            expand=False,
        )
        main_table.add_column("", width=40)
        main_table.add_column("", width=40)
        
        # Process categories in pairs for 2-column layout
        cat_order = ["kpi", "fleet", "engagement", "stealth", "safety", "outcome", "train", "time", "rollout"]
        active_cats = [c for c in cat_order if c in categories]
        
        # Pair up categories
        pairs = []
        for i in range(0, len(active_cats), 2):
            left = active_cats[i] if i < len(active_cats) else None
            right = active_cats[i + 1] if i + 1 < len(active_cats) else None
            pairs.append((left, right))
        
        for left_cat, right_cat in pairs:
            left_panel = self._make_category_panel(left_cat, categories.get(left_cat, {})) if left_cat else ""
            right_panel = self._make_category_panel(right_cat, categories.get(right_cat, {})) if right_cat else ""
            main_table.add_row(left_panel, right_panel)
        
        # Add uncategorized if any
        if uncategorized:
            uncat_panel = self._make_category_panel("other", uncategorized, title="📎 Other", style="dim")
            main_table.add_row(uncat_panel, "")
        
        self.console.print(main_table)
    
    def _make_category_panel(
        self, category: str, metrics: Dict[str, Any], title: Optional[str] = None, style: Optional[str] = None
    ) -> Panel:
        """Create a Rich panel for a metrics category."""
        if category in self.CATEGORIES:
            cat_title, cat_style, border_style = self.CATEGORIES[category]
        else:
            cat_title = title or category.capitalize()
            cat_style = style or "white"
            border_style = "dim"
        
        # Build metrics table
        table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        table.add_column("Metric", style="dim", width=22)
        table.add_column("Value", justify="right", width=12)
        
        for metric, value in metrics.items():
            # Format value
            formatted = self._format_value(metric, value)
            # Clean metric name
            clean_metric = metric.replace("_", " ").title()
            table.add_row(clean_metric, formatted)
        
        return Panel(
            table,
            title=f"[{cat_style}]{cat_title}[/]",
            border_style=border_style,
            padding=(0, 1),
        )
    
    def _format_value(self, metric: str, value: Any) -> str:
        """Format a metric value with appropriate coloring and precision."""
        if isinstance(value, bool):
            return "[green]✓[/]" if value else "[red]✗[/]"
        elif isinstance(value, float):
            # Color based on metric name hints
            if "rate" in metric or "pct" in metric:
                # Percentage-like values
                color = "green" if value >= 0.5 else ("yellow" if value >= 0.2 else "red")
                return f"[{color}]{value:.1%}[/]"
            elif "loss" in metric:
                return f"[yellow]{value:.4f}[/]"
            elif "time" in metric or "elapsed" in metric:
                return f"[dim]{value:.1f}[/]"
            elif value >= 1000:
                return f"[cyan]{value:,.0f}[/]"
            elif value < 0.01 and value != 0:
                return f"[dim]{value:.2e}[/]"
            else:
                return f"{value:.3g}"
        elif isinstance(value, int):
            if value >= 1000:
                return f"[cyan]{value:,}[/]"
            return str(value)
        else:
            return str(value)
    
    def _write_simple(self, key_values: Dict[str, Any], step: int) -> None:
        """Fallback simple output without Rich."""
        print(f"\n--- Step {step} ---")
        for key, value in sorted(key_values.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    def close(self) -> None:
        """Cleanup."""
        pass


class RichTrainingCallback(BaseCallback):
    """
    Callback that prints training statistics with Rich formatting.
    
    Use this with verbose=0 on the model to replace SB3's default dashed-box output
    with beautiful Rich tables.
    """
    
    def __init__(self, print_freq: int = 1, verbose: int = 0):
        """
        Args:
            print_freq: Print every N training iterations (default: every iteration)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.print_freq = print_freq
        self.console = Console() if RICH_AVAILABLE else None
        self.iteration_count = 0
        self.training_start_time: Optional[datetime] = None
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.training_start_time = datetime.now()
        if self.console and RICH_AVAILABLE:
            self.console.print()
            self.console.print("[bold cyan]━━━ Training Started ━━━[/]")
            self.console.print()
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout - this is when SB3 normally prints stats."""
        self.iteration_count += 1
        
        if self.iteration_count % self.print_freq != 0:
            return
        
        if not RICH_AVAILABLE or not self.console:
            return
        
        # Get training info from the model
        try:
            # Access the logger's recorded values
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                values = dict(self.model.logger.name_to_value)
            else:
                values = {}
            
            # Calculate FPS and timing
            elapsed_sec = (datetime.now() - self.training_start_time).total_seconds() if self.training_start_time else 1
            fps = self.num_timesteps / max(1, elapsed_sec)
            
            # Add timing info to values
            values["_fps"] = fps
            values["_elapsed"] = elapsed_sec
            
            # Build training stats table
            self._print_training_stats(values)
            
        except Exception as e:
            # Fallback - just print basic info
            self.console.print(f"[dim]Step {self.num_timesteps:,}[/]")
    
    def _print_training_stats(self, values: Dict[str, Any]) -> None:
        """Print training statistics in a beautiful Rich format."""
        
        # Create compact training panel
        table = Table(
            title=f"[bold cyan]🧠 Training Update[/] [dim]#{self.iteration_count}[/]",
            box=box.ROUNDED,
            border_style="cyan",
            show_header=True,
            header_style="bold",
            padding=(0, 1),
            expand=False,
        )
        
        table.add_column("Metric", style="dim", width=18)
        table.add_column("Value", justify="right", width=12)
        table.add_column("Metric", style="dim", width=18)
        table.add_column("Value", justify="right", width=12)
        
        # Key metrics to show - use our calculated values
        fps = values.get("_fps", values.get("time/fps", 0))
        elapsed = values.get("_elapsed", values.get("time/time_elapsed", 0))
        
        # Format elapsed time nicely
        if elapsed >= 3600:
            elapsed_str = f"{elapsed/3600:.1f}h"
        elif elapsed >= 60:
            elapsed_str = f"{elapsed/60:.1f}m"
        else:
            elapsed_str = f"{elapsed:.0f}s"
        
        metrics = [
            ("timesteps", self.num_timesteps, "iterations", self.iteration_count),
            ("fps", f"{fps:.0f}", "elapsed", elapsed_str),
        ]
        
        # Add training metrics if available
        if "train/policy_gradient_loss" in values:
            pg_loss = values.get("train/policy_gradient_loss", 0)
            vf_loss = values.get("train/value_loss", 0)
            entropy = values.get("train/entropy_loss", 0)
            kl = values.get("train/approx_kl", 0)
            clip = values.get("train/clip_fraction", 0)
            explained_var = values.get("train/explained_variance", 0)
            
            # Color-code based on values
            kl_color = "green" if kl < 0.02 else ("yellow" if kl < 0.05 else "red")
            ev_color = "green" if explained_var > 0.8 else ("yellow" if explained_var > 0.5 else "red")
            
            metrics.extend([
                ("policy_loss", f"{pg_loss:.4f}", "value_loss", f"{vf_loss:.2f}"),
                ("entropy", f"{entropy:.2f}", f"[{kl_color}]approx_kl[/]", f"[{kl_color}]{kl:.4f}[/]"),
                ("clip_frac", f"{clip:.1%}", f"[{ev_color}]expl_var[/]", f"[{ev_color}]{explained_var:.2%}[/]"),
            ])
        
        for m1, v1, m2, v2 in metrics:
            # Format values
            if isinstance(v1, (int, float)) and not isinstance(v1, bool):
                if isinstance(v1, float):
                    v1_str = f"{v1:,.2f}" if v1 >= 100 else f"{v1:.4f}"
                else:
                    v1_str = f"{v1:,}"
            else:
                v1_str = str(v1)
            
            if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                if isinstance(v2, float):
                    v2_str = f"{v2:,.2f}" if v2 >= 100 else f"{v2:.4f}"
                else:
                    v2_str = f"{v2:,}"
            else:
                v2_str = str(v2)
            
            table.add_row(m1, v1_str, m2, v2_str)
        
        self.console.print(table)
    
    def _on_step(self) -> bool:
        return True


def configure_rich_logger(model, suppress_categories: Optional[List[str]] = None):
    """
    Configure a SB3 model to use Rich output format.
    
    NOTE: For best results, set verbose=0 on your model and add RichTrainingCallback
    to your callbacks list instead.
    
    Usage:
        model = PPO(..., verbose=0)
        callbacks.append(RichTrainingCallback())
    
    Args:
        model: SB3 model (PPO, A2C, etc.)
        suppress_categories: Categories to hide from console output
    """
    from stable_baselines3.common.logger import Logger, KVWriter
    
    class RichKVWriter(KVWriter):
        """Adapter for Rich output to SB3's KVWriter interface."""
        
        def __init__(self, suppress_categories: Optional[List[str]] = None):
            self.rich_output = RichOutputFormat(suppress_categories=suppress_categories)
        
        def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, bool], step: int) -> None:
            self.rich_output.write(key_values, key_excluded, step)
        
        def close(self) -> None:
            self.rich_output.close()
    
    # Get existing logger and add Rich output
    if hasattr(model, "logger") and model.logger is not None:
        # Replace stdout writer with Rich writer
        new_output_formats = []
        for writer in model.logger.output_formats:
            # Keep TensorBoard and other file writers, replace human output
            writer_type = type(writer).__name__
            if writer_type == "HumanOutputFormat":
                # Replace with Rich
                new_output_formats.append(RichKVWriter(suppress_categories))
            else:
                new_output_formats.append(writer)
        
        model.logger.output_formats = new_output_formats
