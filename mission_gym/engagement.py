"""Engagement (tag) mechanics."""

from dataclasses import dataclass, field
from typing import Optional
import math

from mission_gym.config import EngagementConfig
from mission_gym.dynamics import UnitState, DynamicsEngine


@dataclass
class TagResult:
    """Result of a tag attempt."""
    success: bool
    attacker_id: int
    target_id: int
    damage_dealt: float = 0.0
    target_disabled: bool = False


@dataclass
class EngagementStats:
    """Tracks engagement statistics per step/episode."""
    tag_attempts_attacker: int = 0
    tag_hits_attacker: int = 0
    tag_attempts_defender: int = 0
    tag_hits_defender: int = 0
    
    def reset(self) -> None:
        """Reset step-level stats."""
        self.tag_attempts_attacker = 0
        self.tag_hits_attacker = 0
        self.tag_attempts_defender = 0
        self.tag_hits_defender = 0


class EngagementSystem:
    """Handles tag/disable mechanics."""
    
    def __init__(self, config: EngagementConfig, dynamics: DynamicsEngine):
        self.config = config
        self.dynamics = dynamics
        self.stats = EngagementStats()
    
    def _calculate_damage(self, distance: float) -> float:
        """
        Calculate damage based on distance with falloff.
        
        - At or below optimal_range: full damage
        - Between optimal_range and max_range: linear falloff to min_damage
        - Beyond max_range: no damage (shouldn't happen as we check range)
        """
        optimal = self.config.tag_optimal_range
        max_range = self.config.tag_range
        full_damage = self.config.tag_damage
        min_damage = self.config.tag_min_damage
        
        if distance <= optimal:
            return full_damage
        elif distance >= max_range:
            return min_damage
        else:
            # Linear interpolation between optimal and max range
            t = (distance - optimal) / (max_range - optimal)
            return full_damage + t * (min_damage - full_damage)
    
    def attempt_tag(
        self,
        attacker: UnitState,
        targets: list[UnitState],
    ) -> Optional[TagResult]:
        """
        Attempt to tag a target.
        
        Args:
            attacker: The attacking unit
            targets: List of potential targets
        
        Returns:
            TagResult if tag was attempted, None if on cooldown or disabled
        """
        # Check if TAG is enabled
        if not self.config.tag_enabled:
            return None
        
        if attacker.is_disabled:
            return None
        
        if attacker.tag_cooldown > 0:
            return None
        
        # Record attempt based on team
        if attacker.team == "attacker":
            self.stats.tag_attempts_attacker += 1
        else:
            self.stats.tag_attempts_defender += 1
        
        # Find closest valid target
        best_target = None
        best_dist = float('inf')
        
        for target in targets:
            if target.is_disabled:
                continue
            if target.team == attacker.team:
                continue
            
            # Check range
            dist = attacker.distance_to(target.x, target.y)
            if dist > self.config.tag_range:
                continue
            
            # Check FOV
            rel_bearing = attacker.relative_bearing_to(target.x, target.y)
            if abs(rel_bearing) > self.config.tag_fov:
                continue
            
            # Check LOS
            if self.config.tag_requires_los:
                has_los = self.dynamics.check_line_of_sight(
                    attacker.x, attacker.y, target.x, target.y,
                    attacker.altitude, target.altitude
                )
                if not has_los:
                    continue
            
            if dist < best_dist:
                best_dist = dist
                best_target = target
        
        # Apply cooldown regardless of hit
        attacker.tag_cooldown = self.config.tag_cooldown
        
        if best_target is None:
            return TagResult(
                success=False,
                attacker_id=attacker.unit_id,
                target_id=-1,
            )
        
        # Record hit based on team
        if attacker.team == "attacker":
            self.stats.tag_hits_attacker += 1
        else:
            self.stats.tag_hits_defender += 1
        
        # Calculate damage with range falloff
        damage = self._calculate_damage(best_dist)
        
        # Apply damage
        old_integrity = best_target.integrity
        best_target.integrity = max(0.0, best_target.integrity - damage)
        damage_dealt = old_integrity - best_target.integrity
        
        # Check if disabled
        target_disabled = False
        if best_target.integrity <= self.config.disabled_threshold:
            best_target.is_disabled = True
            target_disabled = True
        
        return TagResult(
            success=True,
            attacker_id=attacker.unit_id,
            target_id=best_target.unit_id,
            damage_dealt=damage_dealt,
            target_disabled=target_disabled,
        )
    
    def start_scan(self, unit: UnitState, scan_duration: float) -> bool:
        """
        Start a scan action.
        
        Returns:
            True if scan started, False if on cooldown or disabled
        """
        # Check if SCAN is enabled
        if not self.config.scan_enabled:
            return False
        
        if unit.is_disabled:
            return False
        
        if unit.scan_cooldown > 0:
            return False
        
        unit.scan_active = scan_duration
        unit.scan_cooldown = self.config.scan_cooldown
        return True
    
    def get_degraded_mobility(self, unit: UnitState) -> float:
        """Get mobility multiplier based on integrity."""
        if unit.integrity <= self.config.mobility_threshold:
            return 0.5
        return 1.0
    
    def get_degraded_sensor_range(self, unit: UnitState) -> float:
        """Get sensor range multiplier based on integrity."""
        if unit.integrity <= self.config.sensor_threshold:
            return 0.5
        return 1.0
