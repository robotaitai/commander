#!/usr/bin/env python3
"""Training monitoring with HTML dashboard generation."""

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


class HTMLMonitorCallback(BaseCallback):
    """
    Callback that generates an HTML dashboard for monitoring training progress.
    
    Updates an HTML file with:
    - Training stats (timesteps, episodes, FPS)
    - Reward curves
    - Episode length trends
    - Evaluation results
    - Recent episode summaries
    """
    
    def __init__(
        self,
        html_path: str = "training_dashboard.html",
        update_freq: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.html_path = Path(html_path)
        self.update_freq = update_freq
        
        # Training history
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.timesteps_history: List[int] = []
        self.mean_rewards: List[float] = []
        self.eval_rewards: List[float] = []
        self.eval_timesteps: List[int] = []
        
        # Current episode tracking
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
        import yaml
        
        configs = {}
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
    
    def _generate_config_html(self) -> str:
        """Generate HTML for config viewer."""
        if not self.config_yaml:
            return ""
        
        tabs = []
        panels = []
        
        for i, (filename, content) in enumerate(self.config_yaml.items()):
            # Clean filename for ID
            file_id = filename.replace('.', '_').replace('/', '_')
            active = "active" if i == 0 else ""
            
            # Create tab button
            short_name = filename.replace('.yaml', '').replace('_', ' ').title()
            tabs.append(f'<button class="config-tab {active}" onclick="showConfig(\'{file_id}\')">{short_name}</button>')
            
            # Create content panel - escape HTML entities
            escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            display = "block" if i == 0 else "none"
            panels.append(f'<pre class="config-content" id="{file_id}" style="display: {display};">{escaped_content}</pre>')
        
        return f'''
        <div class="chart-container">
            <div class="chart-title">⚙️ Training Configuration</div>
            <div class="config-tabs">
                {"".join(tabs)}
            </div>
            <div class="config-panels">
                {"".join(panels)}
            </div>
        </div>
        '''
    
    def _on_training_start(self) -> None:
        self.start_time = datetime.now()
        self._generate_html()
    
    def _on_step(self) -> bool:
        # Track rewards
        rewards = self.locals.get("rewards", [0])
        dones = self.locals.get("dones", [False])
        
        for reward, done in zip(rewards, dones):
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.episodes_completed += 1
                
                # Track mean reward every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    self.mean_rewards.append(mean_reward)
                    self.timesteps_history.append(self.num_timesteps)
                
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
        
        # Update HTML periodically
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
            self.stored_snapshots = snapshots  # Store for future HTML generations
        self._generate_html()
    
    def _generate_snapshot_gallery(self, snapshots: List[Dict]) -> str:
        """Generate HTML for simulation snapshot gallery."""
        if not snapshots:
            return '<div class="chart-container"><div class="chart-title">🎬 Simulation Snapshots</div><p style="color: var(--text-secondary);">Snapshots will appear here after first evaluation...</p></div>'
        
        sections = []
        for snap in snapshots[-3:]:  # Show last 3 evaluations
            timestep = snap.get("timestep", 0)
            reward = snap.get("reward", 0)
            frames = snap.get("frames", [])
            
            if not frames:
                continue
            
            frame_html = ""
            for frame in frames:
                img_b64 = frame.get("image", "")
                step = frame.get("step", 0)
                r = frame.get("reward", 0)
                frame_html += f'''
                <div class="snapshot-card">
                    <img src="data:image/png;base64,{img_b64}" alt="Step {step}">
                    <div class="snapshot-info">Step {step} | R: {r:.1f}</div>
                </div>'''
            
            sections.append(f'''
            <div class="eval-section">
                <div class="eval-header">
                    <span class="eval-title">Eval @ {timestep:,} steps</span>
                    <span class="eval-reward">Reward: {reward:.2f}</span>
                </div>
                <div class="snapshot-gallery">{frame_html}</div>
            </div>''')
        
        if not sections:
            return ""
        
        return f'''
        <div class="chart-container">
            <div class="chart-title">🎬 Simulation Snapshots (from Evaluations)</div>
            {"".join(sections)}
        </div>'''
    
    def _generate_html(self) -> None:
        """Generate the HTML dashboard."""
        elapsed = datetime.now() - self.start_time if self.start_time else None
        elapsed_str = str(elapsed).split('.')[0] if elapsed else "N/A"
        
        # Calculate stats
        fps = self.num_timesteps / elapsed.total_seconds() if elapsed else 0
        mean_reward_100 = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        mean_length_100 = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        
        # Recent episodes table
        recent_episodes = list(zip(
            range(max(0, len(self.episode_rewards) - 10), len(self.episode_rewards)),
            self.episode_rewards[-10:],
            self.episode_lengths[-10:]
        ))[::-1]
        
        # Chart data as JSON
        chart_data = {
            "timesteps": self.timesteps_history,
            "mean_rewards": self.mean_rewards,
            "eval_timesteps": self.eval_timesteps,
            "eval_rewards": self.eval_rewards,
        }
        
        # Generate snapshot gallery HTML
        snapshot_html = self._generate_snapshot_gallery(self.stored_snapshots)
        
        # Generate config viewer HTML
        config_html = self._generate_config_html()
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="30">
    <title>Mission Gym Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ color: var(--text-secondary); margin-bottom: 2rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 12px;
            padding: 1.5rem;
        }}
        .stat-label {{ color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.5rem; }}
        .stat-value {{ font-size: 1.8rem; font-weight: 600; }}
        .stat-value.positive {{ color: var(--success); }}
        .stat-value.warning {{ color: var(--warning); }}
        .chart-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .chart-title {{ font-size: 1.1rem; margin-bottom: 1rem; color: var(--text-primary); }}
        canvas {{ max-height: 300px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
        }}
        th, td {{ padding: 1rem; text-align: left; border-bottom: 1px solid var(--bg-tertiary); }}
        th {{ background: var(--bg-tertiary); color: var(--text-secondary); font-weight: 500; }}
        tr:hover {{ background: var(--bg-tertiary); }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .badge-success {{ background: rgba(63, 185, 80, 0.2); color: var(--success); }}
        .badge-warning {{ background: rgba(210, 153, 34, 0.2); color: var(--warning); }}
        .last-update {{ color: var(--text-secondary); font-size: 0.85rem; margin-top: 2rem; text-align: center; }}
        .snapshot-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .snapshot-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            overflow: hidden;
        }}
        .snapshot-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .snapshot-info {{
            padding: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        .eval-section {{
            margin-bottom: 2rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 12px;
        }}
        .eval-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--bg-tertiary);
        }}
        .eval-title {{ font-size: 0.9rem; color: var(--text-primary); }}
        .eval-reward {{ font-size: 1.2rem; font-weight: 600; color: var(--accent); }}
        .config-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .config-tab {{
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 6px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        .config-tab:hover {{
            background: var(--bg-primary);
            color: var(--text-primary);
        }}
        .config-tab.active {{
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }}
        .config-content {{
            background: var(--bg-primary);
            border: 1px solid var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
            margin: 0;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.8rem;
            line-height: 1.5;
            color: var(--text-primary);
            white-space: pre-wrap;
        }}
        .config-content::-webkit-scrollbar {{
            width: 8px;
        }}
        .config-content::-webkit-scrollbar-track {{
            background: var(--bg-tertiary);
            border-radius: 4px;
        }}
        .config-content::-webkit-scrollbar-thumb {{
            background: var(--accent);
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Mission Gym Training</h1>
        <p class="subtitle">Real-time training progress dashboard</p>
        
        <div class="grid">
            <div class="stat-card">
                <div class="stat-label">Total Timesteps</div>
                <div class="stat-value">{self.num_timesteps:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Episodes Completed</div>
                <div class="stat-value">{self.episodes_completed:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Reward (100 ep)</div>
                <div class="stat-value {'positive' if mean_reward_100 > 0 else 'warning'}">{mean_reward_100:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Length (100 ep)</div>
                <div class="stat-value">{mean_length_100:.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Training FPS</div>
                <div class="stat-value">{fps:.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Elapsed Time</div>
                <div class="stat-value" style="font-size: 1.2rem;">{elapsed_str}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📈 Reward Progress</div>
            <canvas id="rewardChart"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📊 Recent Episodes</div>
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
                    {"".join(f'''
                    <tr>
                        <td>#{ep+1}</td>
                        <td>{reward:.2f}</td>
                        <td>{length}</td>
                        <td><span class="badge {'badge-success' if reward > 0 else 'badge-warning'}">
                            {'Good' if reward > 0 else 'Learning'}
                        </span></td>
                    </tr>''' for ep, reward, length in recent_episodes)}
                </tbody>
            </table>
        </div>
        
        {snapshot_html}
        
        {config_html}
        
        <p class="last-update">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (auto-refreshes every 30s)</p>
    </div>
    
    <script>
        // Config tab switching
        function showConfig(id) {{
            document.querySelectorAll('.config-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.config-tab').forEach(el => el.classList.remove('active'));
            document.getElementById(id).style.display = 'block';
            event.target.classList.add('active');
        }}
        const data = {json.dumps(chart_data)};
        
        new Chart(document.getElementById('rewardChart'), {{
            type: 'line',
            data: {{
                labels: data.timesteps,
                datasets: [{{
                    label: 'Mean Reward (100 ep)',
                    data: data.mean_rewards,
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    fill: true,
                    tension: 0.3,
                }}, {{
                    label: 'Eval Reward',
                    data: data.eval_timesteps.map((t, i) => ({{ x: t, y: data.eval_rewards[i] }})),
                    borderColor: '#3fb950',
                    backgroundColor: '#3fb950',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    showLine: false,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#c9d1d9' }} }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Timesteps', color: '#8b949e' }},
                        ticks: {{ color: '#8b949e' }},
                        grid: {{ color: '#21262d' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Reward', color: '#8b949e' }},
                        ticks: {{ color: '#8b949e' }},
                        grid: {{ color: '#21262d' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        self.html_path.write_text(html)
        if self.verbose > 0:
            print(f"Dashboard updated: {self.html_path}")


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
        
        # Snapshot storage (base64 encoded images)
        self.snapshots: List[Dict] = []
        self.max_snapshots = 8  # Keep last N evaluation snapshots
    
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
                
                # Capture snapshot at specific steps (first episode only)
                if ep_idx == 0 and self.capture_snapshots and step in self.snapshot_steps:
                    frame = self._capture_frame(step, total_reward, info)
                    if frame:
                        captured_frames.append(frame)
                
                step += 1
            
            episode_rewards.append(total_reward)
        
        mean_reward = np.mean(episode_rewards)
        
        if self.verbose > 0:
            print(f"Eval at {self.num_timesteps}: mean_reward={mean_reward:.2f}")
        
        # Store snapshots
        if captured_frames:
            self.snapshots.append({
                "timestep": self.num_timesteps,
                "reward": mean_reward,
                "frames": captured_frames,
            })
            # Keep only recent snapshots
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
        
        # Update HTML monitor with snapshots
        self.html_monitor.add_eval_result(mean_reward, self.num_timesteps, self.snapshots)
    
    def _capture_frame(self, step: int, reward: float, info: dict) -> Optional[Dict]:
        """Capture a frame from the environment as base64."""
        if not PYGAME_AVAILABLE:
            if self.verbose > 0:
                print(f"Skipping frame capture: PYGAME_AVAILABLE={PYGAME_AVAILABLE}")
            return None
        
        try:
            import pygame
            
            # Initialize pygame if needed (for offscreen rendering)
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
            
            # Create offscreen surface
            screen = pygame.Surface((size, size))
            
            # Render
            self._render_to_surface(screen, env, size, step, reward)
            
            # Convert to PNG bytes
            png_bytes = io.BytesIO()
            pygame.image.save(screen, png_bytes, "PNG")
            png_bytes.seek(0)
            
            # Encode as base64
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
        
        # Colors
        BG = (30, 30, 40)
        GRID = (50, 50, 60)
        OBSTACLE = (80, 80, 100)
        OBJECTIVE = (60, 120, 60)
        OBJECTIVE_CAPTURED = (100, 200, 100)
        ATTACKER_UGV = (100, 150, 255)
        ATTACKER_UAV = (150, 200, 255)
        DEFENDER = (255, 100, 100)
        DISABLED = (100, 100, 100)
        TEXT = (220, 220, 220)
        
        config = env.config
        world_w = config.world.width
        world_h = config.world.height
        scale = size / max(world_w, world_h)
        
        def w2s(x, y):
            return int(x * scale), int((world_h - y) * scale)
        
        screen.fill(BG)
        
        # Grid
        for x in range(0, int(world_w) + 1, 20):
            pygame.draw.line(screen, GRID, w2s(x, 0), w2s(x, world_h), 1)
        for y in range(0, int(world_h) + 1, 20):
            pygame.draw.line(screen, GRID, w2s(0, y), w2s(world_w, y), 1)
        
        # Obstacles
        for obs in config.world.obstacles:
            if obs.type == "circle":
                pygame.draw.circle(screen, OBSTACLE, w2s(obs.x, obs.y), int(obs.radius * scale))
            elif obs.type == "rect":
                corners = obs.get_corners()
                if corners:
                    pygame.draw.polygon(screen, OBSTACLE, [w2s(x, y) for x, y in corners])
        
        # Objective
        obj = env.objective
        progress = obj.capture_progress / obj.capture_time_required
        color = tuple(int(OBJECTIVE[i] + (OBJECTIVE_CAPTURED[i] - OBJECTIVE[i]) * progress) for i in range(3))
        pygame.draw.circle(screen, color, w2s(obj.x, obj.y), int(obj.radius * scale))
        pygame.draw.circle(screen, TEXT, w2s(obj.x, obj.y), int(obj.radius * scale), 2)
        
        # Units
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
        
        # HUD
        font = pygame.font.Font(None, 20)
        screen.blit(font.render(f"Step: {step}", True, TEXT), (5, 5))
        screen.blit(font.render(f"R: {reward:.1f}", True, TEXT), (5, 22))
