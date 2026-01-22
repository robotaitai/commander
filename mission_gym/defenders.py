"""Scripted defender AI with domain randomization."""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import math
import numpy as np

from mission_gym.dynamics import UnitState, get_action_list
from mission_gym.sensors import SensorSystem, Detection


class DefenderMode(Enum):
    """Defender behavior modes."""
    PATROL = "patrol"
    GUARD_OBJECTIVE = "guard_objective"
    INTERCEPT = "intercept"


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
    
    # Domain randomization per episode
    mode: DefenderMode = DefenderMode.PATROL
    reaction_delay: float = 0.0
    time_since_last_decision: float = 0.0
    pending_action: Optional[str] = None


@dataclass
class DefenderRandomizationConfig:
    """Configuration for defender domain randomization."""
    mode_probs: dict[str, float] = field(default_factory=lambda: {
        "patrol": 0.5,
        "guard_objective": 0.3,
        "intercept": 0.2,
    })
    
    # Reaction delays
    delay_min: float = 0.0
    delay_max: float = 0.5
    
    # Epsilon-random
    p_random_action: float = 0.05
    
    # Jitter
    aim_std: float = 5.0  # degrees
    
    # Patrol randomization
    patrol_jitter_enabled: bool = True
    patrol_jitter_radius: float = 10.0


class DefenderController:
    """
    Controls all defender units with scripted behavior and domain randomization.
    
    Behavior modes:
    1. PATROL: Standard patrol around objective using waypoints
    2. GUARD_OBJECTIVE: Stay close to objective, minimal movement
    3. INTERCEPT: Aggressively chase any detected attackers
    """
    
    def __init__(
        self,
        sensor_system: SensorSystem,
        randomization_config: Optional[DefenderRandomizationConfig] = None,
        objective_pos: tuple[float, float] = (100.0, 100.0),
        rng: Optional[np.random.Generator] = None,
    ):
        self.sensor_system = sensor_system
        self.randomization_config = randomization_config or DefenderRandomizationConfig()
        self.objective_pos = objective_pos
        self.rng = rng or np.random.default_rng()
        self.defender_ais: dict[int, DefenderAI] = {}
        self.base_waypoints: dict[int, list[tuple[float, float]]] = {}
    
    def initialize(
        self,
        defenders: list[UnitState],
        patrol_waypoints: list[list[tuple[float, float]]],
    ) -> None:
        """Initialize AI state for all defenders."""
        self.defender_ais = {}
        self.base_waypoints = {}
        
        for i, defender in enumerate(defenders):
            base_waypoints = patrol_waypoints[i] if i < len(patrol_waypoints) else []
            self.base_waypoints[defender.unit_id] = base_waypoints
            
            # Randomize waypoints if enabled
            if self.randomization_config.patrol_jitter_enabled:
                waypoints = self._randomize_waypoints(base_waypoints)
            else:
                waypoints = base_waypoints
            
            behavior = defender.type_config.behavior if defender.type_config else {}
            
            # Sample mode for this defender for this episode
            mode = self._sample_mode()
            
            # Sample reaction delay for this defender for this episode
            reaction_delay = self.rng.uniform(
                self.randomization_config.delay_min,
                self.randomization_config.delay_max,
            )
            
            self.defender_ais[defender.unit_id] = DefenderAI(
                defender_id=defender.unit_id,
                patrol_waypoints=waypoints,
                memory_duration=behavior.get("memory_duration", 5.0),
                engagement_range=behavior.get("engagement_range", 20.0),
                patrol_speed=behavior.get("patrol_speed", 4.0),
                chase_speed=behavior.get("chase_speed", 6.0),
                mode=mode,
                reaction_delay=reaction_delay,
            )
    
    def reset(self) -> None:
        """Reset all AI states."""
        for ai in self.defender_ais.values():
            ai.current_waypoint_idx = 0
            ai.memories = {}
            ai.time_since_last_decision = 0.0
            ai.pending_action = None
            
            # Re-randomize mode and delay for new episode
            ai.mode = self._sample_mode()
            ai.reaction_delay = self.rng.uniform(
                self.randomization_config.delay_min,
                self.randomization_config.delay_max,
            )
            
            # Re-randomize waypoints if enabled
            if self.randomization_config.patrol_jitter_enabled:
                base_waypoints = self.base_waypoints.get(ai.defender_id, [])
                ai.patrol_waypoints = self._randomize_waypoints(base_waypoints)
    
    def _sample_mode(self) -> DefenderMode:
        """Sample a behavior mode based on probabilities."""
        modes = list(self.randomization_config.mode_probs.keys())
        probs = list(self.randomization_config.mode_probs.values())
        chosen = self.rng.choice(modes, p=probs)
        return DefenderMode(chosen)
    
    def _randomize_waypoints(
        self, base_waypoints: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Add jitter to patrol waypoints."""
        if not base_waypoints:
            return []
        
        jittered = []
        radius = self.randomization_config.patrol_jitter_radius
        
        for x, y in base_waypoints:
            dx = self.rng.uniform(-radius, radius)
            dy = self.rng.uniform(-radius, radius)
            jittered.append((x + dx, y + dy))
        
        return jittered
    
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
            
            # Update time since last decision
            ai.time_since_last_decision += dt
            
            # Check if we need to make a new decision (reaction delay)
            if ai.pending_action is None or ai.time_since_last_decision >= ai.reaction_delay:
                # Epsilon-random action
                if self.rng.random() < self.randomization_config.p_random_action:
                    category = defender.category
                    available_actions = get_action_list(category)
                    action = self.rng.choice(available_actions)
                else:
                    # Decide action based on mode
                    action = self._decide_action(defender, ai, attackers, detections)
                
                ai.pending_action = action
                ai.time_since_last_decision = 0.0
            
            actions.append(ai.pending_action)
        
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
        """Decide the action for a defender based on current mode."""
        if ai.mode == DefenderMode.GUARD_OBJECTIVE:
            return self._guard_objective_action(defender, ai, attackers, detections)
        elif ai.mode == DefenderMode.INTERCEPT:
            return self._intercept_action(defender, ai, attackers, detections)
        else:  # PATROL
            return self._patrol_action(defender, ai, attackers, detections)
    
    def _guard_objective_action(
        self,
        defender: UnitState,
        ai: DefenderAI,
        attackers: list[UnitState],
        detections: list[Detection],
    ) -> str:
        """Guard objective mode: stay close, engage if attackers nearby."""
        # Check for attackers in tag range
        visible_attackers = [d for d in detections if d.target_team == "attacker"]
        
        for detection in visible_attackers:
            if detection.range <= ai.engagement_range:
                if defender.tag_cooldown <= 0:
                    return self._aim_and_tag(defender, detection)
        
        # Stay near objective
        obj_x, obj_y = self.objective_pos
        dist_to_obj = defender.distance_to(obj_x, obj_y)
        
        # If too far from objective, move back
        if dist_to_obj > 20.0:
            return self._move_toward(defender, obj_x, obj_y, ai.patrol_speed)
        
        # Otherwise hold position
        return "HOLD"
    
    def _intercept_action(
        self,
        defender: UnitState,
        ai: DefenderAI,
        attackers: list[UnitState],
        detections: list[Detection],
    ) -> str:
        """Intercept mode: aggressively chase and engage attackers."""
        visible_attackers = [d for d in detections if d.target_team == "attacker"]
        
        # If in tag range, tag immediately
        for detection in visible_attackers:
            if detection.range <= ai.engagement_range:
                if defender.tag_cooldown <= 0:
                    return self._aim_and_tag(defender, detection)
        
        # Find target to chase (prefer visible, then memory)
        target_x, target_y = None, None
        
        if visible_attackers:
            # Chase closest visible attacker
            closest = min(visible_attackers, key=lambda d: d.range)
            target_x, target_y = closest.x, closest.y
        elif ai.memories:
            # Chase freshest memory
            freshest = min(ai.memories.values(), key=lambda m: m.time_since_seen)
            target_x, target_y = freshest.last_x, freshest.last_y
        
        # Chase at high speed
        if target_x is not None:
            return self._move_toward(defender, target_x, target_y, ai.chase_speed * 1.2)
        
        # No target, default to patrol
        return self._patrol(defender, ai)
    
    def _patrol_action(
        self,
        defender: UnitState,
        ai: DefenderAI,
        attackers: list[UnitState],
        detections: list[Detection],
    ) -> str:
        """Standard patrol mode: patrol and engage if attackers in range."""
        visible_attackers = [d for d in detections if d.target_team == "attacker"]
        
        # Check for attackers in tag range
        for detection in visible_attackers:
            if detection.range <= ai.engagement_range:
                if defender.tag_cooldown <= 0:
                    return self._aim_and_tag(defender, detection)
        
        # Find target to chase (from memory or current detection)
        target_x, target_y = None, None
        
        # Prefer currently visible attackers
        if visible_attackers:
            closest = min(visible_attackers, key=lambda d: d.range)
            target_x, target_y = closest.x, closest.y
        # Otherwise use memory
        elif ai.memories:
            freshest = min(ai.memories.values(), key=lambda m: m.time_since_seen)
            target_x, target_y = freshest.last_x, freshest.last_y
        
        # If we have a target, move toward it
        if target_x is not None:
            return self._move_toward(defender, target_x, target_y, ai.chase_speed)
        
        # No target, patrol
        return self._patrol(defender, ai)
    
    def _aim_and_tag(self, defender: UnitState, detection: Detection) -> str:
        """Aim at target and tag, with jitter."""
        category = defender.category
        
        # Add aim jitter
        rel_bearing = detection.bearing
        jitter = self.rng.normal(0, self.randomization_config.aim_std)
        rel_bearing += jitter
        
        # Need to turn?
        if abs(rel_bearing) > 15:
            if rel_bearing > 0:
                return "TURN_LEFT" if category == "ground" else "YAW_LEFT"
            else:
                return "TURN_RIGHT" if category == "ground" else "YAW_RIGHT"
        else:
            return "TAG"
    
    def _move_toward(
        self,
        defender: UnitState,
        target_x: float,
        target_y: float,
        target_speed: float,
    ) -> str:
        """Get action to move toward a target with jitter."""
        category = defender.category
        
        rel_bearing = defender.relative_bearing_to(target_x, target_y)
        
        # Add heading jitter
        jitter = self.rng.normal(0, self.randomization_config.aim_std)
        rel_bearing += jitter
        
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
