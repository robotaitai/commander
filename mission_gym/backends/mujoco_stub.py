"""MuJoCo backend placeholder."""

from typing import Any

from mission_gym.backends.base import PhysicsBackend
from mission_gym.dynamics import UnitState


class MuJoCoBackend(PhysicsBackend):
    """
    MuJoCo physics backend placeholder.
    
    This is a stub for future integration with MuJoCo
    for physics simulation with contact dynamics.
    """
    
    def __init__(self):
        raise NotImplementedError(
            "MuJoCoBackend is a placeholder for future MuJoCo integration. "
            "Use Simple2p5DBackend for now."
        )
    
    def initialize(self, config: Any) -> None:
        """Initialize the backend with configuration."""
        raise NotImplementedError()
    
    def reset(self, attackers: list[UnitState], defenders: list[UnitState]) -> None:
        """Reset the simulation with initial unit states."""
        raise NotImplementedError()
    
    def step(
        self,
        attacker_actions: list[str],
        defender_actions: list[str],
        dt: float,
    ) -> tuple[list[UnitState], list[UnitState], list[bool]]:
        """Step the simulation forward."""
        raise NotImplementedError()
    
    def get_unit_states(self) -> tuple[list[UnitState], list[UnitState]]:
        """Get current unit states."""
        raise NotImplementedError()
    
    def close(self) -> None:
        """Clean up backend resources."""
        raise NotImplementedError()
