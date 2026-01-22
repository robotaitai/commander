"""Scenario management - spawning and objective tracking."""

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from mission_gym.config import ScenarioConfig, UnitTypeConfig
from mission_gym.dynamics import UnitState


@dataclass
class ObjectiveState:
    """State of the objective zone."""
    x: float
    y: float
    radius: float
    capture_progress: float = 0.0
    capture_time_required: float = 20.0
    is_captured: bool = False


class ScenarioManager:
    """Manages scenario state including spawning and objective."""
    
    def __init__(
        self,
        config: ScenarioConfig,
        attacker_types: dict[str, UnitTypeConfig],
        defender_types: dict[str, UnitTypeConfig],
        randomization_config: Optional['ScenarioRandomizationConfig'] = None,
        rng: Optional['np.random.Generator'] = None,
    ):
        import numpy as np
        
        self.config = config
        self.attacker_types = attacker_types
        self.defender_types = defender_types
        self.randomization_config = randomization_config
        self.rng = rng or np.random.default_rng()
        
        # Base objective position (will be randomized in reset if enabled)
        self.base_objective_x = config.objective.x
        self.base_objective_y = config.objective.y
        
        # Create objective
        self.objective = ObjectiveState(
            x=config.objective.x,
            y=config.objective.y,
            radius=config.objective.radius,
            capture_time_required=config.objective.capture_time_required,
        )
    
    def spawn_units(self) -> tuple[list[UnitState], list[UnitState]]:
        """
        Spawn all units according to scenario config with optional randomization.
        
        Returns:
            Tuple of (attackers, defenders)
        """
        attackers = []
        defenders = []
        
        unit_id = 0
        
        # Spawn attackers
        for spawn in self.config.attackers:
            type_config = self.attacker_types.get(spawn.unit_type)
            if type_config is None:
                continue
            
            # Apply spawn jitter if enabled
            x, y = spawn.x, spawn.y
            if self.randomization_config and self.randomization_config.spawn_enabled:
                x += self.rng.uniform(
                    -self.randomization_config.attacker_jitter_x,
                    self.randomization_config.attacker_jitter_x
                )
                y += self.rng.uniform(
                    -self.randomization_config.attacker_jitter_y,
                    self.randomization_config.attacker_jitter_y
                )
            
            state = UnitState(
                x=x,
                y=y,
                heading=spawn.heading,
                speed=type_config.initial_speed,  # Start with configured speed
                altitude=spawn.altitude,
                target_altitude=spawn.altitude,
                integrity=type_config.initial_integrity,
                unit_id=unit_id,
                unit_type=spawn.unit_type,
                category=type_config.category,
                team="attacker",
                type_config=type_config,
            )
            attackers.append(state)
            unit_id += 1
        
        # Spawn defenders
        for spawn in self.config.defenders:
            type_config = self.defender_types.get(spawn.unit_type)
            if type_config is None:
                continue
            
            # Apply spawn jitter if enabled
            x, y = spawn.x, spawn.y
            if self.randomization_config and self.randomization_config.spawn_enabled:
                x += self.rng.uniform(
                    -self.randomization_config.defender_jitter_x,
                    self.randomization_config.defender_jitter_x
                )
                y += self.rng.uniform(
                    -self.randomization_config.defender_jitter_y,
                    self.randomization_config.defender_jitter_y
                )
            
            state = UnitState(
                x=x,
                y=y,
                heading=spawn.heading,
                altitude=spawn.altitude,
                target_altitude=spawn.altitude,
                integrity=type_config.initial_integrity,
                unit_id=unit_id,
                unit_type=spawn.unit_type,
                category=type_config.category,
                team="defender",
                type_config=type_config,
            )
            defenders.append(state)
            unit_id += 1
        
        return attackers, defenders
    
    def reset(self) -> tuple[list[UnitState], list[UnitState], ObjectiveState]:
        """Reset the scenario and return fresh state with optional randomization."""
        # Randomize objective position if enabled
        obj_x, obj_y = self.base_objective_x, self.base_objective_y
        if self.randomization_config and self.randomization_config.objective_enabled:
            obj_x += self.rng.uniform(
                -self.randomization_config.objective_jitter_x,
                self.randomization_config.objective_jitter_x
            )
            obj_y += self.rng.uniform(
                -self.randomization_config.objective_jitter_y,
                self.randomization_config.objective_jitter_y
            )
        
        self.objective = ObjectiveState(
            x=obj_x,
            y=obj_y,
            radius=self.config.objective.radius,
            capture_time_required=self.config.objective.capture_time_required,
        )
        
        attackers, defenders = self.spawn_units()
        return attackers, defenders, self.objective
    
    def update_capture(
        self, attackers: list[UnitState], dt: float
    ) -> tuple[float, bool]:
        """
        Update capture progress based on attacker positions.
        
        Args:
            attackers: List of attacker states
            dt: Time delta
        
        Returns:
            Tuple of (capture_progress_delta, is_captured)
        """
        # Check if any non-disabled attacker is in the zone
        any_in_zone = False
        for attacker in attackers:
            if attacker.is_disabled:
                continue
            dist = math.sqrt(
                (attacker.x - self.objective.x) ** 2 +
                (attacker.y - self.objective.y) ** 2
            )
            if dist <= self.objective.radius:
                any_in_zone = True
                break
        
        progress_delta = 0.0
        if any_in_zone:
            progress_delta = dt
            self.objective.capture_progress += dt
            
            if self.objective.capture_progress >= self.objective.capture_time_required:
                self.objective.is_captured = True
        
        return progress_delta, self.objective.is_captured
    
    def get_attacker_patrol_waypoints(self, defender_idx: int) -> list[tuple[float, float]]:
        """Get patrol waypoints for a defender."""
        if defender_idx < len(self.config.defenders):
            return self.config.defenders[defender_idx].patrol_waypoints
        return []
