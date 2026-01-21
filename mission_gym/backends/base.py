"""Abstract base class for physics backends."""

from abc import ABC, abstractmethod
from typing import Any

from mission_gym.dynamics import UnitState


class PhysicsBackend(ABC):
    """Abstract base class for physics simulation backends."""
    
    @abstractmethod
    def initialize(self, config: Any) -> None:
        """Initialize the backend with configuration."""
        pass
    
    @abstractmethod
    def reset(self, attackers: list[UnitState], defenders: list[UnitState]) -> None:
        """Reset the simulation with initial unit states."""
        pass
    
    @abstractmethod
    def step(
        self,
        attacker_actions: list[str],
        defender_actions: list[str],
        dt: float,
    ) -> tuple[list[UnitState], list[UnitState], list[bool]]:
        """
        Step the simulation forward.
        
        Args:
            attacker_actions: Actions for each attacker
            defender_actions: Actions for each defender
            dt: Time delta
        
        Returns:
            Tuple of (updated_attackers, updated_defenders, collision_flags)
        """
        pass
    
    @abstractmethod
    def get_unit_states(self) -> tuple[list[UnitState], list[UnitState]]:
        """Get current unit states."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up backend resources."""
        pass
