"""Scripted defender AI."""

from dataclasses import dataclass, field
from typing import Optional
import math

from mission_gym.dynamics import UnitState, get_action_list
from mission_gym.sensors import SensorSystem, Detection


@dataclass
class DefenderMemory:
    """Memory of a seen attacker."""
    attacker_id: int
    last_x: float
    last_y: float
    time_since_seen: float = 0.0


@dataclass
class DefenderAI:
    """AI state for a single defender."""
    defender_id: int
    patrol_waypoints: list[tuple[float, float]]
    current_waypoint_idx: int = 0
    memories: dict[int, DefenderMemory] = field(default_factory=dict)
    memory_duration: float = 5.0
    engagement_range: float = 20.0
    patrol_speed: float = 4.0
    chase_speed: float = 6.0


class DefenderController:
    """
    Controls all defender units with scripted behavior.
    
    Behavior:
    1. Patrol around objective using waypoints
    2. If any attacker visible and in range, attempt TAG
    3. Otherwise, move toward nearest attacker last-seen position
    """
    
    def __init__(self, sensor_system: SensorSystem):
        self.sensor_system = sensor_system
        self.defender_ais: dict[int, DefenderAI] = {}
    
    def initialize(
        self,
        defenders: list[UnitState],
        patrol_waypoints: list[list[tuple[float, float]]],
    ) -> None:
        """Initialize AI state for all defenders."""
        self.defender_ais = {}
        for i, defender in enumerate(defenders):
            waypoints = patrol_waypoints[i] if i < len(patrol_waypoints) else []
            behavior = defender.type_config.behavior if defender.type_config else {}
            
            self.defender_ais[defender.unit_id] = DefenderAI(
                defender_id=defender.unit_id,
                patrol_waypoints=waypoints,
                memory_duration=behavior.get("memory_duration", 5.0),
                engagement_range=behavior.get("engagement_range", 20.0),
                patrol_speed=behavior.get("patrol_speed", 4.0),
                chase_speed=behavior.get("chase_speed", 6.0),
            )
    
    def reset(self) -> None:
        """Reset all AI states."""
        for ai in self.defender_ais.values():
            ai.current_waypoint_idx = 0
            ai.memories = {}
    
    def get_actions(
        self,
        defenders: list[UnitState],
        attackers: list[UnitState],
        dt: float,
    ) -> list[str]:
        """
        Get actions for all defenders.
        
        Args:
            defenders: Current defender states
            attackers: Current attacker states (for detection)
            dt: Time delta
        
        Returns:
            List of action strings for each defender
        """
        actions = []
        
        for defender in defenders:
            if defender.is_disabled:
                actions.append("NOOP")
                continue
            
            ai = self.defender_ais.get(defender.unit_id)
            if ai is None:
                actions.append("NOOP")
                continue
            
            # Update memories with detections
            detections = self.sensor_system.get_detections(defender, attackers)
            self._update_memories(ai, detections, dt)
            
            # Decide action
            action = self._decide_action(defender, ai, attackers, detections)
            actions.append(action)
        
        return actions
    
    def _update_memories(
        self,
        ai: DefenderAI,
        detections: list[Detection],
        dt: float,
    ) -> None:
        """Update memory based on current detections."""
        # Age all memories
        for memory in list(ai.memories.values()):
            memory.time_since_seen += dt
            if memory.time_since_seen > ai.memory_duration:
                del ai.memories[memory.attacker_id]
        
        # Update with new detections
        for detection in detections:
            if detection.target_team == "attacker":
                ai.memories[detection.target_id] = DefenderMemory(
                    attacker_id=detection.target_id,
                    last_x=detection.x,
                    last_y=detection.y,
                    time_since_seen=0.0,
                )
    
    def _decide_action(
        self,
        defender: UnitState,
        ai: DefenderAI,
        attackers: list[UnitState],
        detections: list[Detection],
    ) -> str:
        """Decide the action for a defender."""
        category = defender.category
        actions = get_action_list(category)
        
        # Check for attackers in tag range
        visible_attackers = [d for d in detections if d.target_team == "attacker"]
        
        for detection in visible_attackers:
            if detection.range <= ai.engagement_range:
                # Check if we can tag (cooldown is handled by engagement)
                if defender.tag_cooldown <= 0:
                    # Point toward target first
                    rel_bearing = detection.bearing
                    if abs(rel_bearing) > 15:  # Need to turn
                        if rel_bearing > 0:
                            return "TURN_LEFT" if category == "ground" else "YAW_LEFT"
                        else:
                            return "TURN_RIGHT" if category == "ground" else "YAW_RIGHT"
                    else:
                        return "TAG"
        
        # Find target to chase (from memory or current detection)
        target_x, target_y = None, None
        
        # Prefer currently visible attackers
        if visible_attackers:
            closest = min(visible_attackers, key=lambda d: d.range)
            target_x, target_y = closest.x, closest.y
        # Otherwise use memory
        elif ai.memories:
            # Find freshest memory
            freshest = min(ai.memories.values(), key=lambda m: m.time_since_seen)
            target_x, target_y = freshest.last_x, freshest.last_y
        
        # If we have a target, move toward it
        if target_x is not None:
            return self._move_toward(defender, target_x, target_y, ai.chase_speed)
        
        # No target, patrol
        return self._patrol(defender, ai)
    
    def _move_toward(
        self,
        defender: UnitState,
        target_x: float,
        target_y: float,
        target_speed: float,
    ) -> str:
        """Get action to move toward a target."""
        category = defender.category
        
        rel_bearing = defender.relative_bearing_to(target_x, target_y)
        dist = defender.distance_to(target_x, target_y)
        
        # If close enough, hold
        if dist < 2.0:
            return "HOLD"
        
        # Need to turn?
        if abs(rel_bearing) > 20:
            if rel_bearing > 0:
                return "TURN_LEFT" if category == "ground" else "YAW_LEFT"
            else:
                return "TURN_RIGHT" if category == "ground" else "YAW_RIGHT"
        
        # Adjust speed
        if defender.speed < target_speed * 0.9:
            return "THROTTLE_UP"
        elif defender.speed > target_speed * 1.1:
            return "THROTTLE_DOWN"
        
        return "NOOP"
    
    def _patrol(self, defender: UnitState, ai: DefenderAI) -> str:
        """Get action for patrol behavior."""
        if not ai.patrol_waypoints:
            return "HOLD"
        
        # Get current waypoint
        waypoint = ai.patrol_waypoints[ai.current_waypoint_idx]
        target_x, target_y = waypoint
        
        dist = defender.distance_to(target_x, target_y)
        
        # Check if we've reached the waypoint
        if dist < 3.0:
            ai.current_waypoint_idx = (ai.current_waypoint_idx + 1) % len(ai.patrol_waypoints)
            waypoint = ai.patrol_waypoints[ai.current_waypoint_idx]
            target_x, target_y = waypoint
        
        return self._move_toward(defender, target_x, target_y, ai.patrol_speed)
