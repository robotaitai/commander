"""Reward function for the environment."""

from dataclasses import dataclass
import math
import numpy as np

from mission_gym.config import RewardConfig, ObjectiveConfig
from mission_gym.dynamics import UnitState


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
    """Calculates rewards based on configuration."""
    
    def __init__(self, config: RewardConfig, objective: ObjectiveConfig):
        self.config = config
        self.objective = objective
    
    def calculate(
        self,
        attackers: list[UnitState],
        step_info: StepInfo,
        prev_distances: list[float],
    ) -> tuple[float, dict]:
        """
        Calculate the reward for a step.
        
        Args:
            attackers: Current attacker states
            step_info: Information about what happened
            prev_distances: Previous distances to objective for each attacker
        
        Returns:
            Total reward and info dict
        """
        reward = 0.0
        info = {}
        
        # Capture progress reward
        if step_info.capture_progress_delta > 0:
            capture_reward = self.config.capture_progress * step_info.capture_progress_delta
            reward += capture_reward
            info["capture_reward"] = capture_reward
        
        # Win bonus
        if step_info.won:
            reward += self.config.win_bonus
            info["win_bonus"] = self.config.win_bonus
        
        # Time penalty
        reward += self.config.time_penalty
        info["time_penalty"] = self.config.time_penalty
        
        # Collision penalty
        if step_info.collisions > 0:
            collision_penalty = self.config.collision_penalty * step_info.collisions
            reward += collision_penalty
            info["collision_penalty"] = collision_penalty
        
        # Integrity loss penalty
        if step_info.integrity_lost > 0:
            integrity_penalty = self.config.integrity_loss_penalty * step_info.integrity_lost
            reward += integrity_penalty
            info["integrity_penalty"] = integrity_penalty
        
        # Unit disabled penalty
        if step_info.units_disabled > 0:
            disabled_penalty = self.config.unit_disabled_penalty * step_info.units_disabled
            reward += disabled_penalty
            info["disabled_penalty"] = disabled_penalty
        
        # Detected penalty (optional)
        if self.config.enable_detected_penalty and step_info.any_detected:
            reward += self.config.detected_time_penalty
            info["detected_penalty"] = self.config.detected_time_penalty
        
        # Shaping: approach objective
        approach_reward = 0.0
        for i, attacker in enumerate(attackers):
            if attacker.is_disabled:
                continue
            curr_dist = math.sqrt(
                (attacker.x - self.objective.x) ** 2 +
                (attacker.y - self.objective.y) ** 2
            )
            if i < len(prev_distances):
                dist_delta = prev_distances[i] - curr_dist
                approach_reward += self.config.approach_objective * dist_delta
        
        if approach_reward != 0:
            reward += approach_reward
            info["approach_reward"] = approach_reward
        
        # Shaping: spread formation (encourage units to spread out)
        spread_reward = 0.0
        active_attackers = [a for a in attackers if not a.is_disabled]
        if len(active_attackers) > 1:
            min_pair_dist = float('inf')
            for i, a1 in enumerate(active_attackers):
                for a2 in active_attackers[i+1:]:
                    dist = math.sqrt((a1.x - a2.x) ** 2 + (a1.y - a2.y) ** 2)
                    min_pair_dist = min(min_pair_dist, dist)
            
            # Reward for keeping minimum distance > 5 meters
            if min_pair_dist > 5.0:
                spread_reward = self.config.spread_formation
                reward += spread_reward
                info["spread_reward"] = spread_reward
        
        info["total_reward"] = reward
        return reward, info
    
    def get_distances_to_objective(self, attackers: list[UnitState]) -> list[float]:
        """Get distances from each attacker to the objective."""
        distances = []
        for attacker in attackers:
            dist = math.sqrt(
                (attacker.x - self.objective.x) ** 2 +
                (attacker.y - self.objective.y) ** 2
            )
            distances.append(dist)
        return distances
