"""Episode metrics tracking for Mission Gym.

Tracks detailed KPIs for each episode including:
- Mission outcomes (win/loss, termination reason)
- Fleet performance (distance, speed, formation)
- Engagement stats (tag attempts, hits, integrity)
- Detection/stealth metrics
- Collisions and constraint violations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class EpisodeMetrics:
    """Container for all episode metrics."""
    
    # Mission outcome (top-level KPIs)
    win: bool = False
    terminated_reason: str = ""  # "captured", "timeout", "all_disabled"
    episode_steps: int = 0
    episode_sim_time: float = 0.0
    
    # Objective tracking
    final_capture_progress: float = 0.0
    time_to_capture: Optional[float] = None
    time_in_objective_zone: float = 0.0  # cumulative seconds with ≥1 attacker in zone
    
    # Fleet status
    num_attackers_alive_end: int = 0
    num_attackers_disabled_total: int = 0
    
    # Collisions / constraint violations
    collisions_total: int = 0
    out_of_bounds_events: int = 0
    
    # Integrity tracking
    integrity_lost_total: float = 0.0
    
    # Detection / stealth metrics
    detected_time: float = 0.0  # seconds where any attacker is detected
    first_detect_time: Optional[float] = None
    detection_events: int = 0  # rising edges (newly detected)
    
    # Engagement stats (tag/disable mechanics)
    tag_attempts_attacker: int = 0
    tag_hits_attacker: int = 0
    tag_attempts_defender: int = 0
    tag_hits_defender: int = 0
    disable_events: int = 0
    first_disable_time: Optional[float] = None
    
    # Per-unit tracking
    distance_by_unit: List[float] = field(default_factory=list)
    integrity_lost_by_unit: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary with derived rates."""
        d = {}
        
        # Copy all fields
        d["win"] = self.win
        d["terminated_reason"] = self.terminated_reason
        d["episode_steps"] = self.episode_steps
        d["episode_sim_time"] = self.episode_sim_time
        d["final_capture_progress"] = self.final_capture_progress
        d["time_to_capture"] = self.time_to_capture
        d["time_in_objective_zone"] = self.time_in_objective_zone
        d["num_attackers_alive_end"] = self.num_attackers_alive_end
        d["num_attackers_disabled_total"] = self.num_attackers_disabled_total
        d["collisions_total"] = self.collisions_total
        d["out_of_bounds_events"] = self.out_of_bounds_events
        d["integrity_lost_total"] = self.integrity_lost_total
        d["detected_time"] = self.detected_time
        d["first_detect_time"] = self.first_detect_time
        d["detection_events"] = self.detection_events
        d["tag_attempts_attacker"] = self.tag_attempts_attacker
        d["tag_hits_attacker"] = self.tag_hits_attacker
        d["tag_attempts_defender"] = self.tag_attempts_defender
        d["tag_hits_defender"] = self.tag_hits_defender
        d["disable_events"] = self.disable_events
        d["first_disable_time"] = self.first_disable_time
        
        # Derived rates
        d["tag_hit_rate_attacker"] = (
            self.tag_hits_attacker / self.tag_attempts_attacker
            if self.tag_attempts_attacker > 0 else 0.0
        )
        d["tag_hit_rate_defender"] = (
            self.tag_hits_defender / self.tag_attempts_defender
            if self.tag_attempts_defender > 0 else 0.0
        )
        
        # Aggregate per-unit metrics
        d["distance_total"] = sum(self.distance_by_unit) if self.distance_by_unit else 0.0
        d["distance_mean"] = (
            d["distance_total"] / len(self.distance_by_unit)
            if self.distance_by_unit else 0.0
        )
        
        # Detection percentage
        d["detected_time_pct"] = (
            self.detected_time / self.episode_sim_time * 100
            if self.episode_sim_time > 0 else 0.0
        )
        
        return d


class MetricsTracker:
    """
    Tracks per-episode metrics during environment execution.
    
    Usage:
        tracker = MetricsTracker(num_attackers=4)
        tracker.reset()
        
        # In step loop:
        tracker.record_step(...)
        
        # When episode ends:
        metrics_dict = tracker.finish(win=True, reason="captured")
    """
    
    def __init__(self, num_attackers: int):
        self.num_attackers = num_attackers
        self.reset()
    
    def reset(self) -> None:
        """Reset metrics for a new episode."""
        self.ep = EpisodeMetrics(
            distance_by_unit=[0.0] * self.num_attackers,
            integrity_lost_by_unit=[0.0] * self.num_attackers,
        )
        self._prev_positions: Optional[List[tuple]] = None
        self._prev_any_detected: bool = False
        self._prev_integrities: Optional[List[float]] = None
    
    def record_step(
        self,
        attackers: list,
        objective,
        dt: float,
        collisions: int,
        integrity_lost: float,
        units_disabled: int,
        any_detected: bool,
        in_objective_zone: bool,
        tag_attempts_attacker: int = 0,
        tag_hits_attacker: int = 0,
        tag_attempts_defender: int = 0,
        tag_hits_defender: int = 0,
    ) -> None:
        """
        Record metrics for a single step.
        
        Args:
            attackers: List of attacker UnitState objects
            objective: ObjectiveState with capture_progress
            dt: Time delta for this step
            collisions: Number of collisions this step
            integrity_lost: Total integrity lost this step
            units_disabled: Number of units newly disabled this step
            any_detected: Whether any attacker is currently detected
            in_objective_zone: Whether any attacker is in objective zone
            tag_attempts_attacker: TAG actions by attackers this step
            tag_hits_attacker: Successful TAG hits by attackers this step
            tag_attempts_defender: TAG actions by defenders this step
            tag_hits_defender: Successful TAG hits by defenders this step
        """
        self.ep.episode_steps += 1
        self.ep.episode_sim_time += dt
        
        # Collisions
        self.ep.collisions_total += collisions
        
        # Integrity tracking
        self.ep.integrity_lost_total += integrity_lost
        
        # Disable events
        if units_disabled > 0:
            self.ep.disable_events += units_disabled
            if self.ep.first_disable_time is None:
                self.ep.first_disable_time = self.ep.episode_sim_time
        
        # Time in objective zone
        if in_objective_zone:
            self.ep.time_in_objective_zone += dt
        
        # Detection timing + events
        if any_detected:
            self.ep.detected_time += dt
            if self.ep.first_detect_time is None:
                self.ep.first_detect_time = self.ep.episode_sim_time
        
        # Rising edge detection (newly detected)
        if any_detected and not self._prev_any_detected:
            self.ep.detection_events += 1
        self._prev_any_detected = any_detected
        
        # Tag stats
        self.ep.tag_attempts_attacker += tag_attempts_attacker
        self.ep.tag_hits_attacker += tag_hits_attacker
        self.ep.tag_attempts_defender += tag_attempts_defender
        self.ep.tag_hits_defender += tag_hits_defender
        
        # Distance travelled per unit
        if self._prev_positions is None:
            self._prev_positions = [(a.x, a.y) for a in attackers]
        else:
            for i, a in enumerate(attackers):
                if i < len(self._prev_positions):
                    px, py = self._prev_positions[i]
                    self.ep.distance_by_unit[i] += math.hypot(a.x - px, a.y - py)
            self._prev_positions = [(a.x, a.y) for a in attackers]
        
        # Per-unit integrity tracking
        if self._prev_integrities is not None:
            for i, a in enumerate(attackers):
                if i < len(self._prev_integrities):
                    loss = max(0, self._prev_integrities[i] - a.integrity)
                    self.ep.integrity_lost_by_unit[i] += loss
        self._prev_integrities = [a.integrity for a in attackers]
        
        # Update capture progress
        self.ep.final_capture_progress = objective.capture_progress
    
    def finish(self, win: bool, reason: str, attackers: list) -> Dict:
        """
        Finalize episode metrics.
        
        Args:
            win: Whether the mission was successful
            reason: Termination reason ("captured", "timeout", "all_disabled")
            attackers: Final list of attacker UnitState objects
        
        Returns:
            Dictionary of all metrics
        """
        self.ep.win = bool(win)
        self.ep.terminated_reason = reason
        
        # Record time to capture if won
        if win and self.ep.time_to_capture is None:
            self.ep.time_to_capture = self.ep.episode_sim_time
        
        # Count alive/disabled attackers at end
        self.ep.num_attackers_alive_end = sum(1 for a in attackers if not a.is_disabled)
        self.ep.num_attackers_disabled_total = sum(1 for a in attackers if a.is_disabled)
        
        return self.ep.to_dict()
