"""
Reward function for the environment.

This module provides backward-compatible access to the reward system while
using the new component-based architecture under the hood.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import math

from mission_gym.config import RewardConfig, ObjectiveConfig
from mission_gym.dynamics import UnitState
from mission_gym.reward_components import (
    RewardRegistry,
    RewardContext,
    RewardResult,
    EpisodeRewardTracker,
    create_default_registry,
)


@dataclass
class StepInfo:
    """Information about what happened in a step."""
    capture_progress_delta: float = 0.0
    collisions: int = 0
    integrity_lost: float = 0.0
    units_disabled: int = 0
    any_detected: bool = False
    won: bool = False


class RewardFunction:
    """
    Calculates rewards based on configuration using the component-based system.
    
    This class provides backward compatibility with the original interface
    while using the new modular reward component architecture.
    """
    
    def __init__(self, config: RewardConfig, objective: ObjectiveConfig):
        self.config = config
        self.objective = objective
        
        # Create component registry with default components
        self.registry = create_default_registry()
        
        # Episode tracker for detailed statistics
        self.tracker = EpisodeRewardTracker()
        
        # Apply weights from config
        self._apply_config_weights()
    
    def _apply_config_weights(self):
        """Apply weights from RewardConfig to components."""
        # All weights are already applied in the config values
        # Components multiply by config.* directly
        weight_map = {
            "capture_progress": self.config.capture_progress,
            "win_bonus": self.config.win_bonus,
            "zone_entry": 1.0,  # Uses config.zone_entry_bonus
            "zone_time": 1.0,   # Uses config.zone_time
            "min_dist_potential": 1.0,  # Uses config.min_dist_potential
            "ring_bonus": 1.0,  # Uses config.ring_bonus
            "time_penalty": 1.0,
            "collision_penalty": 1.0,
            "integrity_loss": 1.0,
            "unit_disabled": 1.0,
            "detected_penalty": 1.0,
            "approach_objective": 1.0,
            "spread_formation": 1.0,
        }
        
        for name, weight in weight_map.items():
            self.registry.set_weight(name, weight)
        
        # Handle detected penalty toggle
        if not self.config.enable_detected_penalty:
            self.registry.enable("detected_penalty", False)
        
        # Disable approach_objective if weight is 0 (using min_dist instead)
        if self.config.approach_objective == 0:
            self.registry.enable("approach_objective", False)
    
    def calculate(
        self,
        attackers: List[UnitState],
        step_info: StepInfo,
        prev_distances: List[float],
        defenders: Optional[List[UnitState]] = None,
    ) -> tuple[float, dict]:
        """
        Calculate the reward for a step.
        
        Args:
            attackers: Current attacker states
            step_info: Information about what happened
            prev_distances: Previous distances to objective for each attacker
            defenders: Optional defender states
        
        Returns:
            Total reward and info dict with component breakdown
        """
        # Build context for reward calculation
        ctx = RewardContext(
            attackers=attackers,
            defenders=defenders or [],
            step_info=step_info,
            prev_distances=prev_distances,
            config=self.config,
            objective=self.objective,
        )
        
        # Calculate all reward components
        results = self.registry.calculate_all(ctx)
        
        # Record in tracker
        self.tracker.record_step(results, step_info)
        
        # Build info dict
        info = {}
        for result in results:
            if result.value != 0:
                info[f"{result.component_name}_reward"] = result.value
        
        total_reward = self.registry.get_total(results)
        info["total_reward"] = total_reward
        
        # Add component breakdown for visualization
        info["_component_breakdown"] = [
            {
                "name": r.component_name,
                "value": r.value,
                "category": r.category,
            }
            for r in results
        ]
        
        return total_reward, info
    
    def get_distances_to_objective(self, attackers: List[UnitState]) -> List[float]:
        """Get distances from each attacker to the objective."""
        distances = []
        for attacker in attackers:
            dist = math.sqrt(
                (attacker.x - self.objective.x) ** 2 +
                (attacker.y - self.objective.y) ** 2
            )
            distances.append(dist)
        return distances
    
    def reset(self) -> None:
        """Reset all reward components for a new episode.
        
        This is important for stateful components like MinDistanceToObjectiveReward
        that track state across steps within an episode.
        """
        self.registry.reset_all_stats()
    
    def end_episode(self) -> dict:
        """End the current episode and return stats."""
        stats = self.tracker.end_episode()
        return stats.to_dict()
    
    def get_component_stats(self) -> dict:
        """Get statistics for all reward components."""
        return self.registry.get_statistics()
    
    def get_aggregate_stats(self) -> dict:
        """Get aggregate statistics across all episodes."""
        return self.tracker.get_aggregate_stats()
    
    def get_component_configs(self) -> list:
        """Get configuration info for all components (for visualization)."""
        return [
            {
                "name": comp.name,
                "category": comp.category.value,
                "color": comp.color,
                "icon": comp.icon,
                "description": comp.description,
                "enabled": comp.enabled,
                "weight": comp.weight,
            }
            for comp in self.registry
        ]
    
    def register_component(self, component) -> None:
        """Register a custom reward component."""
        self.registry.register(component)
    
    def enable_component(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a reward component."""
        return self.registry.enable(name, enabled)
    
    def set_component_weight(self, name: str, weight: float) -> bool:
        """Set the weight of a reward component."""
        return self.registry.set_weight(name, weight)
