"""Main Gymnasium environment for Mission Gym."""

from typing import Any, Optional
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from mission_gym.config import FullConfig
from mission_gym.dynamics import UnitState, get_action_list, get_num_actions
from mission_gym.scenario import ScenarioManager, ObjectiveState
from mission_gym.sensors import SensorSystem
from mission_gym.engagement import EngagementSystem
from mission_gym.reward import RewardFunction, StepInfo
from mission_gym.defenders import DefenderController
from mission_gym.backends.simple2p5d import Simple2p5DBackend
from mission_gym.renderer import Renderer, PYGAME_AVAILABLE
from mission_gym.metrics import MetricsTracker


class MissionGymEnv(gym.Env):
    """
    Mission Gym: A game-like RL environment for fleet command and capture.
    
    Observation Space:
        Dict with:
        - 'bev': Box(0, 1, (128, 128, 8), float32) - Bird's eye view raster
        - 'vec': Box(-inf, inf, (N,), float32) - Vector features
    
    Action Space:
        MultiDiscrete with one action per attacker unit.
    
    Reward:
        Configurable via reward.yaml
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[FullConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        # Load configuration
        self.config = config if config is not None else FullConfig.load()
        
        # Initialize RNG
        self._np_random = np.random.default_rng(seed)
        
        # Initialize scenario manager
        self.scenario = ScenarioManager(
            self.config.scenario,
            self.config.attacker_types,
            self.config.defender_types,
        )
        
        # Initialize backend
        self.backend = Simple2p5DBackend()
        self.backend.initialize(self.config)
        
        # Initialize engagement system
        self.engagement = EngagementSystem(
            self.config.engagement,
            self.backend.dynamics,
        )
        
        # Initialize sensor system
        self.sensors = SensorSystem(
            self.config.sensors,
            self.backend.dynamics,
            self.config.engagement.sensor_threshold,
            self._np_random,
        )
        
        # Initialize reward function
        self.reward_fn = RewardFunction(
            self.config.reward,
            self.config.scenario.objective,
        )
        
        # Initialize defender controller
        self.defender_controller = DefenderController(self.sensors)
        
        # State
        self.attackers: list[UnitState] = []
        self.defenders: list[UnitState] = []
        self.objective: ObjectiveState = None
        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.prev_distances: list[float] = []
        
        # Metrics tracker (initialized properly in reset after we know num_attackers)
        self.metrics: Optional[MetricsTracker] = None
        
        # Early termination tracking
        self._best_min_dist: Optional[float] = None
        self._last_progress_time: float = 0.0
        self._prev_capture_progress: float = 0.0
        
        # Rendering
        self.render_mode = render_mode
        self.renderer: Optional[Renderer] = None
        self.selected_unit: int = 0
        
        # Build action and observation spaces
        self._build_spaces()
    
    def _build_spaces(self) -> None:
        """Build action and observation spaces."""
        # For now, spawn units to get counts
        attackers, defenders, _ = self.scenario.reset()
        self.num_attackers = len(attackers)
        self.num_defenders = len(defenders)
        
        # Action space: MultiDiscrete with one action per attacker
        action_dims = []
        for spawn in self.config.scenario.attackers:
            type_config = self.config.attacker_types.get(spawn.unit_type)
            if type_config:
                num_actions = len(type_config.actions)
                action_dims.append(num_actions)
        
        self.action_space = spaces.MultiDiscrete(action_dims)
        
        # Observation space: VECTOR ONLY (no BEV images for policy)
        # Vector: per-unit features + global features
        # Per unit: x, y, heading_cos, heading_sin, speed, integrity, tag_cd, scan_cd, altitude, disabled = 10
        # Global: time_remaining, capture_progress = 2
        vec_dim = self.num_attackers * 10 + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(vec_dim,),
            dtype=np.float32
        )
        
        # Store BEV shape for debug rendering (not part of observation)
        self._bev_shape = (128, 128, 8)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.sensors.rng = self._np_random
        
        # Reset scenario
        self.attackers, self.defenders, self.objective = self.scenario.reset()
        
        # Reset early termination tracking
        self._best_min_dist = min(
            np.hypot(a.x - self.objective.x, a.y - self.objective.y)
            for a in self.attackers if not a.is_disabled
        ) if self.attackers else float('inf')
        self._last_progress_time = 0.0
        self._prev_capture_progress = 0.0

        # Reset backend
        self.backend.reset(self.attackers, self.defenders)
        
        # Reset defender AI
        patrol_waypoints = [
            self.scenario.get_attacker_patrol_waypoints(i)
            for i in range(len(self.defenders))
        ]
        self.defender_controller.initialize(self.defenders, patrol_waypoints)
        self.defender_controller.reset()
        
        # Reset state
        self.sim_time = 0.0
        self.step_count = 0
        self.prev_distances = self.reward_fn.get_distances_to_objective(self.attackers)
        self.last_actions = ["---"] * len(self.attackers)  # Initialize command display
        
        # Reset reward components (important for stateful components like min_dist_potential)
        self.reward_fn.reset()
        
        # Reset metrics tracker
        if self.metrics is None:
            self.metrics = MetricsTracker(len(self.attackers))
        self.metrics.reset()
        
        # Reset engagement stats
        self.engagement.stats.reset()


        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one environment step.
        
        Args:
            action: MultiDiscrete action array
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Track integrity before step
        prev_integrities = [a.integrity for a in self.attackers]
        prev_disabled = sum(1 for a in self.attackers if a.is_disabled)
        
        # Convert actions to strings
        attacker_actions = self._decode_actions(action)
        self.last_actions = attacker_actions  # Store for rendering
        
        # Get defender actions
        dt = 1.0 / self.config.world.tick_rate
        defender_actions = self.defender_controller.get_actions(
            self.defenders, self.attackers, dt
        )
        
        # Execute action_repeat physics steps
        total_collisions = 0
        total_capture_delta = 0.0
        
        for _ in range(self.config.world.action_repeat):
            # Handle TAG and SCAN actions
            self._process_engagement_actions(attacker_actions, defender_actions)
            
            # Step physics
            self.attackers, self.defenders, collisions = self.backend.step(
                attacker_actions, defender_actions, dt
            )
            total_collisions += sum(1 for c in collisions[:len(self.attackers)] if c)
            
            # Update capture
            capture_delta, captured = self.scenario.update_capture(self.attackers, dt)
            total_capture_delta += capture_delta
            
            self.sim_time += dt
            
            if captured:
                break
        
        self.step_count += 1
        
        # Calculate integrity loss
        integrity_lost = sum(
            max(0, prev - curr.integrity)
            for prev, curr in zip(prev_integrities, self.attackers)
        )
        
        # Count newly disabled units
        curr_disabled = sum(1 for a in self.attackers if a.is_disabled)
        newly_disabled = curr_disabled - prev_disabled
        
        # Check if any attacker is detected
        any_detected = False
        for defender in self.defenders:
            detections = self.sensors.get_detections(defender, self.attackers)
            if any(d.target_team == "attacker" for d in detections):
                any_detected = True
                break
        
        # Build step info
        step_info = StepInfo(
            capture_progress_delta=total_capture_delta,
            collisions=total_collisions,
            integrity_lost=integrity_lost,
            units_disabled=newly_disabled,
            any_detected=any_detected,
            won=self.objective.is_captured,
            tag_hits=self.engagement.stats.tag_hits_attacker,
            defenders=self.defenders,
        )
        
        # Calculate reward
        reward, reward_info = self.reward_fn.calculate(
            self.attackers, step_info, self.prev_distances
        )
        
        # Update distances for next step
        self.prev_distances = self.reward_fn.get_distances_to_objective(self.attackers)
        
        # Check if any attacker is in objective zone
        in_objective_zone = any(
            not a.is_disabled and 
            math.hypot(a.x - self.objective.x, a.y - self.objective.y) <= self.objective.radius
            for a in self.attackers
        )
        
        # Compute tag opportunities for diagnostics
        tag_opportunities = self._count_tag_opportunities()
        
        # Record step metrics (get stats from engagement system)
        self.metrics.record_step(
            attackers=self.attackers,
            defenders=self.defenders,
            objective=self.objective,
            dt=dt * self.config.world.action_repeat,
            collisions=total_collisions,
            integrity_lost=integrity_lost,
            units_disabled=newly_disabled,
            any_detected=any_detected,
            in_objective_zone=in_objective_zone,
            tag_opportunities=tag_opportunities,
            tag_attempts_attacker=self.engagement.stats.tag_attempts_attacker,
            tag_hits_attacker=self.engagement.stats.tag_hits_attacker,
            tag_attempts_defender=self.engagement.stats.tag_attempts_defender,
            tag_hits_defender=self.engagement.stats.tag_hits_defender,
        )
        
        # Reset engagement stats for next step
        self.engagement.stats.reset()
        
        # === Early Termination Logic ===
        # Compute current min distance to objective
        curr_min_dist = min(
            np.hypot(a.x - self.objective.x, a.y - self.objective.y)
            for a in self.attackers if not a.is_disabled
        ) if any(not a.is_disabled for a in self.attackers) else float("inf")
        
        # Check if agent made progress this step
        made_progress = False
        
        # Progress signal 1: Capture progress increased
        if total_capture_delta > 0:
            made_progress = True
        
        # Progress signal 2: Distance improved by at least min_dist_epsilon
        if curr_min_dist < (self._best_min_dist - self.config.termination.min_dist_epsilon):
            made_progress = True
            self._best_min_dist = curr_min_dist
        
        # Update last progress time
        if made_progress:
            self._last_progress_time = self.sim_time
        
        # Check for stagnation (no progress for too long)
        time_since_progress = self.sim_time - self._last_progress_time
        stalled = time_since_progress >= self.config.termination.stagnation_seconds
        
        # Check termination conditions
        terminated = self.objective.is_captured
        outcome = "captured" if terminated else None
        
        # Check for early success (optional)
        if self.config.termination.early_success_capture_progress is not None:
            if self.objective.capture_progress >= self.config.termination.early_success_capture_progress:
                terminated = True
                outcome = "early_success"
        
        # Check truncation (time limit, all disabled, or stalled)
        time_limit = self.sim_time >= self.config.world.max_duration
        all_disabled = all(a.is_disabled for a in self.attackers)
        truncated = time_limit or all_disabled or stalled
        
        # Set outcome for truncated episodes
        if truncated and outcome is None:
            if stalled:
                outcome = "stalled"
            elif all_disabled:
                outcome = "all_disabled"
            else:
                outcome = "timeout"
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = reward_info
        info["outcome"] = outcome
        
        # Pass through component breakdown for monitoring
        if "_component_breakdown" in reward_info:
            info["_component_breakdown"] = reward_info["_component_breakdown"]
        
        # Finalize episode metrics if done
        if terminated or truncated:
            info["episode_metrics"] = self.metrics.finish(
                win=terminated and outcome in ["captured", "early_success"],
                reason=outcome,
                attackers=self.attackers,
            )
        
        return obs, reward, terminated, truncated, info
    
    def _decode_actions(self, action: np.ndarray) -> list[str]:
        """Convert action indices to action strings."""
        actions = []
        for i, act_idx in enumerate(action):
            if i < len(self.attackers):
                attacker = self.attackers[i]
                if attacker.type_config:
                    action_list = attacker.type_config.actions
                    if 0 <= act_idx < len(action_list):
                        actions.append(action_list[act_idx])
                    else:
                        actions.append("NOOP")
                else:
                    actions.append("NOOP")
            else:
                actions.append("NOOP")
        return actions
    
    def _process_engagement_actions(
        self,
        attacker_actions: list[str],
        defender_actions: list[str],
    ) -> None:
        """Process TAG and SCAN actions."""
        all_units = self.attackers + self.defenders
        
        # Attacker engagement
        for i, action in enumerate(attacker_actions):
            if i >= len(self.attackers):
                continue
            attacker = self.attackers[i]
            
            if action == "TAG":
                self.engagement.attempt_tag(attacker, self.defenders)
            elif action == "SCAN":
                self.engagement.start_scan(
                    attacker, self.config.engagement.scan_duration
                )
        
        # Defender engagement
        for i, action in enumerate(defender_actions):
            if i >= len(self.defenders):
                continue
            defender = self.defenders[i]
            
            if action == "TAG":
                self.engagement.attempt_tag(defender, self.attackers)
            elif action == "SCAN":
                self.engagement.start_scan(
                    defender, self.config.engagement.scan_duration
                )
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector (BEV not included for policy)."""
        return self._build_vector()
    
    def get_debug_bev(self) -> np.ndarray:
        """Generate BEV for debugging/rendering (not part of policy observation)."""
        return self._render_bev()
    
    def _render_bev(self) -> np.ndarray:
        """Render bird's eye view raster."""
        bev = np.zeros((128, 128, 8), dtype=np.float32)
        
        scale_x = 128 / self.config.world.width
        scale_y = 128 / self.config.world.height
        
        def world_to_bev(x: float, y: float) -> tuple[int, int]:
            bx = int(x * scale_x)
            by = int((self.config.world.height - y) * scale_y)  # Flip Y
            return max(0, min(127, bx)), max(0, min(127, by))
        
        # Channel 0: Obstacles
        for obs in self.config.world.obstacles:
            if obs.type == "circle":
                cx, cy = world_to_bev(obs.x, obs.y)
                radius = int(obs.radius * scale_x)
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx * dx + dy * dy <= radius * radius:
                            px, py = cx + dx, cy + dy
                            if 0 <= px < 128 and 0 <= py < 128:
                                bev[py, px, 0] = 1.0
            elif obs.type == "rect":
                corners = obs.get_corners()
                if corners:
                    # Simple bounding box fill for rectangles
                    min_x = min(c[0] for c in corners)
                    max_x = max(c[0] for c in corners)
                    min_y = min(c[1] for c in corners)
                    max_y = max(c[1] for c in corners)
                    for wx in range(int(min_x), int(max_x) + 1):
                        for wy in range(int(min_y), int(max_y) + 1):
                            px, py = world_to_bev(wx, wy)
                            bev[py, px, 0] = 1.0
        
        # Channel 1: Objective zone
        ox, oy = world_to_bev(self.objective.x, self.objective.y)
        radius = int(self.objective.radius * scale_x)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = ox + dx, oy + dy
                    if 0 <= px < 128 and 0 <= py < 128:
                        bev[py, px, 1] = 1.0
        
        # Channel 2: Attackers
        for attacker in self.attackers:
            if not attacker.is_disabled:
                ax, ay = world_to_bev(attacker.x, attacker.y)
                radius = 2
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        px, py = ax + dx, ay + dy
                        if 0 <= px < 128 and 0 <= py < 128:
                            bev[py, px, 2] = 1.0
        
        # Channel 3: Defenders
        for defender in self.defenders:
            if not defender.is_disabled:
                dx, dy = world_to_bev(defender.x, defender.y)
                radius = 2
                for ddx in range(-radius, radius + 1):
                    for ddy in range(-radius, radius + 1):
                        px, py = dx + ddx, dy + ddy
                        if 0 <= px < 128 and 0 <= py < 128:
                            bev[py, px, 3] = 1.0
        
        # Channel 4: Attacker type IDs (normalized)
        for i, attacker in enumerate(self.attackers):
            if not attacker.is_disabled:
                ax, ay = world_to_bev(attacker.x, attacker.y)
                if 0 <= ax < 128 and 0 <= ay < 128:
                    bev[ay, ax, 4] = (i + 1) / len(self.attackers)
        
        # Channel 5: Defender FOV (simplified as circles around defenders)
        for defender in self.defenders:
            if not defender.is_disabled:
                dx, dy = world_to_bev(defender.x, defender.y)
                fov_radius = int(20 * scale_x)  # Approximate detection radius
                for ddx in range(-fov_radius, fov_radius + 1):
                    for ddy in range(-fov_radius, fov_radius + 1):
                        if ddx * ddx + ddy * ddy <= fov_radius * fov_radius:
                            px, py = dx + ddx, dy + ddy
                            if 0 <= px < 128 and 0 <= py < 128:
                                bev[py, px, 5] = 0.3
        
        # Channel 6: Tag cooldown heatmap
        for attacker in self.attackers:
            if attacker.tag_cooldown > 0:
                ax, ay = world_to_bev(attacker.x, attacker.y)
                if 0 <= ax < 128 and 0 <= ay < 128:
                    bev[ay, ax, 6] = attacker.tag_cooldown / self.config.engagement.tag_cooldown
        
        # Channel 7: Capture progress (broadcast)
        progress = self.objective.capture_progress / self.objective.capture_time_required
        bev[:, :, 7] = progress
        
        return bev
    
    def _build_vector(self) -> np.ndarray:
        """Build vector observation."""
        features = []
        
        # Per-unit features
        for attacker in self.attackers:
            # Normalize heading from [0, 360) to [-1, 1] using sin/cos
            heading_rad = math.radians(attacker.heading)
            features.extend([
                attacker.x / self.config.world.width,  # Normalize to [0, 1]
                attacker.y / self.config.world.height,
                math.cos(heading_rad),  # Heading as cos component [-1, 1]
                math.sin(heading_rad),  # Heading as sin component [-1, 1]
                attacker.speed / 15.0,  # Normalize by max reasonable speed
                attacker.integrity / 100.0,
                attacker.tag_cooldown / self.config.engagement.tag_cooldown if self.config.engagement.tag_cooldown > 0 else 0.0,
                attacker.scan_cooldown / self.config.engagement.scan_cooldown if self.config.engagement.scan_cooldown > 0 else 0.0,
                float(attacker.altitude) / 2.0,  # Normalize by max altitude
                float(attacker.is_disabled),
            ])
        
        # Global features
        time_remaining = self.config.world.max_duration - self.sim_time
        features.append(time_remaining / self.config.world.max_duration)
        
        capture_progress = self.objective.capture_progress / self.objective.capture_time_required
        features.append(capture_progress)
        
        return np.array(features, dtype=np.float32)
    
    def _count_tag_opportunities(self) -> int:
        """
        Count number of viable tag opportunities this step.
        
        Returns number of (attacker, defender) pairs where:
        - Both units are not disabled
        - Defender is within tag_range of attacker
        - Defender is within tag_fov of attacker
        - LOS exists (if required)
        """
        opportunities = 0
        tag_range = self.config.engagement.tag_range
        tag_fov = self.config.engagement.tag_fov
        requires_los = self.config.engagement.tag_requires_los
        
        for attacker in self.attackers:
            if attacker.is_disabled:
                continue
            
            for defender in self.defenders:
                if defender.is_disabled:
                    continue
                
                # Check range
                dist = attacker.distance_to(defender.x, defender.y)
                if dist > tag_range:
                    continue
                
                # Check FOV
                rel_bearing = attacker.relative_bearing_to(defender.x, defender.y)
                if abs(rel_bearing) > tag_fov:
                    continue
                
                # Check LOS
                if requires_los:
                    has_los = self.backend.dynamics.check_line_of_sight(
                        attacker.x, attacker.y, defender.x, defender.y,
                        attacker.altitude, defender.altitude
                    )
                    if not has_los:
                        continue
                
                opportunities += 1
        
        return opportunities
    
    def _get_info(self) -> dict:
        """Build info dictionary."""
        return {
            "sim_time": self.sim_time,
            "step_count": self.step_count,
            "capture_progress": self.objective.capture_progress,
            "attackers_disabled": sum(1 for a in self.attackers if a.is_disabled),
            "defenders_disabled": sum(1 for d in self.defenders if d.is_disabled),
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.renderer is None:
            if not PYGAME_AVAILABLE:
                return None
            self.renderer = Renderer(self.config)
            self.renderer.initialize()
        
        # Update command display with last actions
        if hasattr(self, 'last_actions') and self.last_actions:
            self.renderer.update_commands(self.last_actions)
        
        time_remaining = self.config.world.max_duration - self.sim_time
        
        self.renderer.render(
            self.attackers,
            self.defenders,
            self.objective,
            time_remaining,
            self.selected_unit,
        )
        
        if self.render_mode == "human":
            self.renderer.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            # Would need to capture the pygame surface as array
            # For now just render and return None
            return None
    
    def close(self) -> None:
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.backend.close()
    
    def set_selected_unit(self, unit_idx: int) -> None:
        """Set the selected unit for rendering."""
        self.selected_unit = max(0, min(unit_idx, len(self.attackers) - 1))
