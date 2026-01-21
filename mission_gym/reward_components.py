"""
Reward Components API - A clean system for defining and tracking reward components.

This module provides a flexible way to define reward components that are:
- Easy to add/modify
- Automatically tracked
- Visualized in the dashboard

Example usage:
    
    # Define a new reward component
    @reward_component(
        name="capture_progress",
        category="objective",
        color="#3fb950",
        description="Reward for progressing objective capture"
    )
    def capture_progress_reward(ctx: RewardContext) -> float:
        if ctx.step_info.capture_progress_delta > 0:
            return ctx.config.capture_progress * ctx.step_info.capture_progress_delta
        return 0.0
    
    # Or use the class-based approach for more complex logic
    class ApproachObjectiveReward(RewardComponent):
        name = "approach_objective"
        category = "shaping"
        color = "#58a6ff"
        description = "Reward for moving toward the objective"
        
        def calculate(self, ctx: RewardContext) -> float:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Any
import math


class RewardCategory(Enum):
    """Categories for reward components - used for grouping in visualization."""
    OBJECTIVE = "objective"      # Goal-related rewards (capture, win)
    PENALTY = "penalty"          # Negative rewards (collision, damage)
    SHAPING = "shaping"          # Intermediate shaping rewards
    BONUS = "bonus"              # Optional bonus rewards
    SURVIVAL = "survival"        # Staying alive, health-related


@dataclass
class RewardContext:
    """Context passed to reward components for calculation."""
    # Unit states
    attackers: List[Any]  # List[UnitState]
    defenders: List[Any]  # List[UnitState]
    
    # Step information
    step_info: Any  # StepInfo object
    
    # Previous step data for delta calculations
    prev_distances: List[float]
    
    # Configuration
    config: Any  # RewardConfig
    objective: Any  # ObjectiveConfig
    
    # Episode state
    timestep: int = 0
    episode_length: int = 0
    
    # Custom data storage for complex rewards
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardResult:
    """Result from a reward component calculation."""
    component_name: str
    value: float
    category: str
    raw_value: float = 0.0  # Before any scaling
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RewardComponentConfig:
    """Configuration for a reward component."""
    name: str
    category: RewardCategory
    color: str = "#58a6ff"  # Hex color for visualization
    description: str = ""
    enabled: bool = True
    weight: float = 1.0  # Multiplier for the component
    icon: str = "📊"  # Emoji icon for display


class RewardComponent(ABC):
    """Base class for reward components."""
    
    # These should be overridden by subclasses
    name: str = "unnamed"
    category: RewardCategory = RewardCategory.SHAPING
    color: str = "#58a6ff"
    description: str = ""
    icon: str = "📊"
    
    def __init__(self, weight: float = 1.0, enabled: bool = True):
        self.weight = weight
        self.enabled = enabled
        self._accumulated_value = 0.0
        self._call_count = 0
    
    @abstractmethod
    def calculate(self, ctx: RewardContext) -> float:
        """Calculate the reward value. Must be implemented by subclasses."""
        pass
    
    def __call__(self, ctx: RewardContext) -> RewardResult:
        """Calculate and return a RewardResult."""
        if not self.enabled:
            return RewardResult(
                component_name=self.name,
                value=0.0,
                category=self.category.value,
                raw_value=0.0,
            )
        
        raw_value = self.calculate(ctx)
        weighted_value = raw_value * self.weight
        
        # Track for statistics
        self._accumulated_value += weighted_value
        self._call_count += 1
        
        return RewardResult(
            component_name=self.name,
            value=weighted_value,
            category=self.category.value,
            raw_value=raw_value,
        )
    
    def reset_stats(self):
        """Reset accumulated statistics."""
        self._accumulated_value = 0.0
        self._call_count = 0
    
    @property
    def average_value(self) -> float:
        """Get average value per call."""
        if self._call_count == 0:
            return 0.0
        return self._accumulated_value / self._call_count
    
    @property
    def total_value(self) -> float:
        """Get total accumulated value."""
        return self._accumulated_value
    
    def get_config(self) -> RewardComponentConfig:
        """Get configuration for this component."""
        return RewardComponentConfig(
            name=self.name,
            category=self.category,
            color=self.color,
            description=self.description,
            enabled=self.enabled,
            weight=self.weight,
            icon=self.icon,
        )


# ============================================================================
# Built-in Reward Components
# ============================================================================

class CaptureProgressReward(RewardComponent):
    """Reward for making progress on capturing the objective."""
    name = "capture_progress"
    category = RewardCategory.OBJECTIVE
    color = "#3fb950"  # Green
    description = "Reward for capturing the objective zone"
    icon = "🎯"
    
    def calculate(self, ctx: RewardContext) -> float:
        if ctx.step_info.capture_progress_delta > 0:
            return ctx.config.capture_progress * ctx.step_info.capture_progress_delta
        return 0.0


class WinBonusReward(RewardComponent):
    """Large bonus reward for winning the episode."""
    name = "win_bonus"
    category = RewardCategory.OBJECTIVE
    color = "#a371f7"  # Purple
    description = "Bonus for successfully completing the mission"
    icon = "🏆"
    
    def calculate(self, ctx: RewardContext) -> float:
        if ctx.step_info.won:
            return ctx.config.win_bonus
        return 0.0


class TimePenaltyReward(RewardComponent):
    """Small per-step time penalty to encourage efficiency."""
    name = "time_penalty"
    category = RewardCategory.PENALTY
    color = "#8b949e"  # Gray
    description = "Small penalty per timestep to encourage speed"
    icon = "⏱️"
    
    def calculate(self, ctx: RewardContext) -> float:
        return ctx.config.time_penalty


class CollisionPenaltyReward(RewardComponent):
    """Penalty for colliding with obstacles."""
    name = "collision_penalty"
    category = RewardCategory.PENALTY
    color = "#f85149"  # Red
    description = "Penalty for hitting obstacles"
    icon = "💥"
    
    def calculate(self, ctx: RewardContext) -> float:
        if ctx.step_info.collisions > 0:
            return ctx.config.collision_penalty * ctx.step_info.collisions
        return 0.0


class IntegrityLossPenaltyReward(RewardComponent):
    """Penalty for losing unit integrity (taking damage)."""
    name = "integrity_loss"
    category = RewardCategory.SURVIVAL
    color = "#f0883e"  # Orange
    description = "Penalty for units taking damage"
    icon = "❤️‍🩹"
    
    def calculate(self, ctx: RewardContext) -> float:
        if ctx.step_info.integrity_lost > 0:
            return ctx.config.integrity_loss_penalty * ctx.step_info.integrity_lost
        return 0.0


class UnitDisabledPenaltyReward(RewardComponent):
    """Large penalty when a unit is disabled."""
    name = "unit_disabled"
    category = RewardCategory.SURVIVAL
    color = "#da3633"  # Dark red
    description = "Penalty when a unit is disabled"
    icon = "💀"
    
    def calculate(self, ctx: RewardContext) -> float:
        if ctx.step_info.units_disabled > 0:
            return ctx.config.unit_disabled_penalty * ctx.step_info.units_disabled
        return 0.0


class DetectedPenaltyReward(RewardComponent):
    """Penalty for being detected by defenders."""
    name = "detected_penalty"
    category = RewardCategory.PENALTY
    color = "#d29922"  # Yellow
    description = "Penalty while any unit is detected"
    icon = "👁️"
    
    def calculate(self, ctx: RewardContext) -> float:
        if ctx.config.enable_detected_penalty and ctx.step_info.any_detected:
            return ctx.config.detected_time_penalty
        return 0.0


class ApproachObjectiveReward(RewardComponent):
    """Shaping reward for moving toward the objective."""
    name = "approach_objective"
    category = RewardCategory.SHAPING
    color = "#58a6ff"  # Blue
    description = "Reward for moving closer to the objective"
    icon = "🧭"
    
    def calculate(self, ctx: RewardContext) -> float:
        approach_reward = 0.0
        for i, attacker in enumerate(ctx.attackers):
            if attacker.is_disabled:
                continue
            curr_dist = math.sqrt(
                (attacker.x - ctx.objective.x) ** 2 +
                (attacker.y - ctx.objective.y) ** 2
            )
            if i < len(ctx.prev_distances):
                dist_delta = ctx.prev_distances[i] - curr_dist
                approach_reward += ctx.config.approach_objective * dist_delta
        return approach_reward


class SpreadFormationReward(RewardComponent):
    """Reward for maintaining spread formation between units."""
    name = "spread_formation"
    category = RewardCategory.SHAPING
    color = "#7ee787"  # Light green
    description = "Reward for keeping units spread apart"
    icon = "↔️"
    
    def calculate(self, ctx: RewardContext) -> float:
        active_attackers = [a for a in ctx.attackers if not a.is_disabled]
        if len(active_attackers) <= 1:
            return 0.0
        
        min_pair_dist = float('inf')
        for i, a1 in enumerate(active_attackers):
            for a2 in active_attackers[i+1:]:
                dist = math.sqrt((a1.x - a2.x) ** 2 + (a1.y - a2.y) ** 2)
                min_pair_dist = min(min_pair_dist, dist)
        
        # Reward for keeping minimum distance > 5 meters
        if min_pair_dist > 5.0:
            return ctx.config.spread_formation
        return 0.0


# ============================================================================
# Reward Registry - Manages all reward components
# ============================================================================

class RewardRegistry:
    """
    Registry that manages all reward components.
    
    Usage:
        registry = RewardRegistry()
        registry.register(CaptureProgressReward())
        registry.register(TimePenaltyReward(weight=0.5))
        
        # Calculate all rewards
        results = registry.calculate_all(context)
        total = sum(r.value for r in results)
    """
    
    def __init__(self):
        self._components: Dict[str, RewardComponent] = {}
        self._order: List[str] = []  # Maintain insertion order
    
    def register(self, component: RewardComponent) -> "RewardRegistry":
        """Register a reward component. Returns self for chaining."""
        self._components[component.name] = component
        if component.name not in self._order:
            self._order.append(component.name)
        return self
    
    def unregister(self, name: str) -> bool:
        """Unregister a component by name. Returns True if found."""
        if name in self._components:
            del self._components[name]
            self._order.remove(name)
            return True
        return False
    
    def get(self, name: str) -> Optional[RewardComponent]:
        """Get a component by name."""
        return self._components.get(name)
    
    def enable(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a component."""
        if name in self._components:
            self._components[name].enabled = enabled
            return True
        return False
    
    def set_weight(self, name: str, weight: float) -> bool:
        """Set the weight of a component."""
        if name in self._components:
            self._components[name].weight = weight
            return True
        return False
    
    def calculate_all(self, ctx: RewardContext) -> List[RewardResult]:
        """Calculate all reward components."""
        results = []
        for name in self._order:
            component = self._components[name]
            result = component(ctx)
            results.append(result)
        return results
    
    def get_total(self, results: List[RewardResult]) -> float:
        """Get total reward from results."""
        return sum(r.value for r in results)
    
    def reset_all_stats(self):
        """Reset statistics for all components."""
        for component in self._components.values():
            component.reset_stats()
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all components."""
        stats = {}
        for name, component in self._components.items():
            stats[name] = {
                "total": component.total_value,
                "average": component.average_value,
                "calls": component._call_count,
                "enabled": component.enabled,
                "weight": component.weight,
                "category": component.category.value,
                "color": component.color,
                "icon": component.icon,
                "description": component.description,
            }
        return stats
    
    def get_component_configs(self) -> List[RewardComponentConfig]:
        """Get configurations for all components."""
        return [self._components[name].get_config() for name in self._order]
    
    @property
    def component_names(self) -> List[str]:
        """Get list of registered component names."""
        return self._order.copy()
    
    def __len__(self) -> int:
        return len(self._components)
    
    def __iter__(self):
        for name in self._order:
            yield self._components[name]


def create_default_registry() -> RewardRegistry:
    """Create a registry with all default reward components."""
    registry = RewardRegistry()
    
    # Objective rewards
    registry.register(CaptureProgressReward())
    registry.register(WinBonusReward())
    
    # Penalties
    registry.register(TimePenaltyReward())
    registry.register(CollisionPenaltyReward())
    registry.register(IntegrityLossPenaltyReward())
    registry.register(UnitDisabledPenaltyReward())
    registry.register(DetectedPenaltyReward())
    
    # Shaping rewards
    registry.register(ApproachObjectiveReward())
    registry.register(SpreadFormationReward())
    
    return registry


# ============================================================================
# Episode Reward Tracker - Tracks rewards across an episode
# ============================================================================

@dataclass
class EpisodeRewardStats:
    """Statistics for a single episode."""
    total_reward: float = 0.0
    component_totals: Dict[str, float] = field(default_factory=dict)
    component_counts: Dict[str, int] = field(default_factory=dict)
    category_totals: Dict[str, float] = field(default_factory=dict)
    step_rewards: List[float] = field(default_factory=list)
    length: int = 0
    won: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_reward": self.total_reward,
            "component_totals": self.component_totals,
            "category_totals": self.category_totals,
            "length": self.length,
            "won": self.won,
            "mean_step_reward": sum(self.step_rewards) / len(self.step_rewards) if self.step_rewards else 0,
        }


class EpisodeRewardTracker:
    """
    Tracks reward components across an episode for detailed analysis.
    
    Usage:
        tracker = EpisodeRewardTracker()
        
        for step in episode:
            results = registry.calculate_all(ctx)
            tracker.record_step(results)
        
        stats = tracker.get_episode_stats()
    """
    
    def __init__(self):
        self._current_episode = EpisodeRewardStats()
        self._episode_history: List[EpisodeRewardStats] = []
        self._max_history = 100
    
    def record_step(self, results: List[RewardResult], step_info: Any = None):
        """Record rewards for a single step."""
        step_total = 0.0
        
        for result in results:
            step_total += result.value
            
            # Track component totals
            if result.component_name not in self._current_episode.component_totals:
                self._current_episode.component_totals[result.component_name] = 0.0
                self._current_episode.component_counts[result.component_name] = 0
            self._current_episode.component_totals[result.component_name] += result.value
            if result.value != 0:
                self._current_episode.component_counts[result.component_name] += 1
            
            # Track category totals
            if result.category not in self._current_episode.category_totals:
                self._current_episode.category_totals[result.category] = 0.0
            self._current_episode.category_totals[result.category] += result.value
        
        self._current_episode.step_rewards.append(step_total)
        self._current_episode.total_reward += step_total
        self._current_episode.length += 1
        
        # Check for win
        if step_info and hasattr(step_info, 'won') and step_info.won:
            self._current_episode.won = True
    
    def end_episode(self) -> EpisodeRewardStats:
        """End current episode and return stats."""
        stats = self._current_episode
        self._episode_history.append(stats)
        
        # Trim history
        if len(self._episode_history) > self._max_history:
            self._episode_history = self._episode_history[-self._max_history:]
        
        # Start new episode
        self._current_episode = EpisodeRewardStats()
        return stats
    
    def get_current_stats(self) -> EpisodeRewardStats:
        """Get stats for current (incomplete) episode."""
        return self._current_episode
    
    def get_history(self) -> List[EpisodeRewardStats]:
        """Get episode history."""
        return self._episode_history
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all episodes."""
        if not self._episode_history:
            return {}
        
        # Aggregate component stats
        component_stats = {}
        category_stats = {}
        
        for ep in self._episode_history:
            for comp_name, total in ep.component_totals.items():
                if comp_name not in component_stats:
                    component_stats[comp_name] = {"total": 0, "episodes": 0, "mean": 0}
                component_stats[comp_name]["total"] += total
                component_stats[comp_name]["episodes"] += 1
            
            for cat, total in ep.category_totals.items():
                if cat not in category_stats:
                    category_stats[cat] = {"total": 0, "episodes": 0}
                category_stats[cat]["total"] += total
                category_stats[cat]["episodes"] += 1
        
        # Calculate means
        for name, stats in component_stats.items():
            stats["mean"] = stats["total"] / stats["episodes"] if stats["episodes"] > 0 else 0
        
        for cat, stats in category_stats.items():
            stats["mean"] = stats["total"] / stats["episodes"] if stats["episodes"] > 0 else 0
        
        return {
            "num_episodes": len(self._episode_history),
            "mean_reward": sum(ep.total_reward for ep in self._episode_history) / len(self._episode_history),
            "mean_length": sum(ep.length for ep in self._episode_history) / len(self._episode_history),
            "win_rate": sum(1 for ep in self._episode_history if ep.won) / len(self._episode_history),
            "component_stats": component_stats,
            "category_stats": category_stats,
        }
