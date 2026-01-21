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
        
        # Episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episodes_completed = 0
        
        # Stats
        self.start_time: Optional[datetime] = None
        self.last_update_timestep = 0
        
        # Stored snapshots from eval callbacks
        self.stored_snapshots: List[Dict] = []
        
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
        
        for reward, done, info in zip(rewards, dones, infos):
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            # Track component breakdown if available
            if "_component_breakdown" in info:
                self.record_component_breakdown(info["_component_breakdown"])
            
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.episodes_completed += 1
                
                if len(self.episode_rewards) % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    self.mean_rewards.append(mean_reward)
                    self.timesteps_history.append(self.num_timesteps)
                
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
        
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
        
        # Calculate win rate (rough estimate from positive rewards)
        if self.episode_rewards:
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
        
        # Chart data
        chart_data = {
            "timesteps": self.timesteps_history[-100:],
            "mean_rewards": self.mean_rewards[-100:],
            "eval_timesteps": self.eval_timesteps,
            "eval_rewards": self.eval_rewards,
            "component_history": {k: v[-50:] for k, v in self.component_history.items()},
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
        
        # Build recent episodes table
        recent_html = self._build_recent_episodes_html()
        
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
        
        .reward-category {{
            margin-left: auto;
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
        
        <!-- Recent Episodes -->
        {recent_html}
        
        <!-- Snapshots -->
        {snapshots_html}
        
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
            
            cards_html += f'''
            <div class="reward-card">
                <div class="reward-header">
                    <span class="reward-icon">{comp["icon"]}</span>
                    <span class="reward-name">{comp["display_name"]}</span>
                    <span class="reward-category {comp["category"]}">{comp["category"]}</span>
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
            frames_html += f'''
            <div class="snapshot-card">
                <img src="data:image/png;base64,{frame.get("image", "")}" alt="Step {frame.get("step", 0)}">
                <div class="snapshot-info">Step {frame.get("step", 0)} • R: {frame.get("reward", 0):.1f}</div>
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
    
    def _build_recent_episodes_html(self) -> str:
        """Build recent episodes table HTML."""
        if not self.episode_rewards:
            return ""
        
        recent = list(zip(
            range(max(0, len(self.episode_rewards) - 10), len(self.episode_rewards)),
            self.episode_rewards[-10:],
            self.episode_lengths[-10:]
        ))[::-1]
        
        rows_html = ""
        for ep, reward, length in recent:
            if reward > 50:
                badge = '<span class="badge badge-success">WIN</span>'
            elif reward > 0:
                badge = '<span class="badge badge-warning">PARTIAL</span>'
            else:
                badge = '<span class="badge badge-danger">LOSS</span>'
            
            rows_html += f'''
            <tr>
                <td>#{ep + 1}</td>
                <td style="font-family: 'JetBrains Mono', monospace;">{reward:.1f}</td>
                <td>{length}</td>
                <td>{badge}</td>
            </tr>'''
        
        return f'''
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">📊</span> Recent Episodes</div>
            </div>
            <div class="panel-body" style="padding: 0;">
                <table>
                    <thead>
                        <tr>
                            <th>Episode</th>
                            <th>Reward</th>
                            <th>Length</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
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
        self.snapshot_steps = snapshot_steps or [0, 5, 15, 30, 50, 100]
        
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
                    frame = self._capture_frame(step, total_reward, info)
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
    
    def _capture_frame(self, step: int, reward: float, info: dict) -> Optional[Dict]:
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
            
            return {
                "step": step,
                "reward": reward,
                "image": b64,
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
    Callback that logs episode metrics to TensorBoard.
    
    Logs key performance indicators from the EpisodeMetrics collected by the env:
    - Mission outcomes (win rate, time to capture)
    - Engagement stats (tag hit rates)
    - Detection/stealth metrics
    - Fleet performance (distance, collisions)
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0
        self.wins = 0
    
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
