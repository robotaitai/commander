"""Simple 2.5D physics backend using custom dynamics."""

from typing import Any

from mission_gym.backends.base import PhysicsBackend
from mission_gym.config import FullConfig
from mission_gym.dynamics import UnitState, DynamicsEngine, get_action_list


class Simple2p5DBackend(PhysicsBackend):
    """
    Simple 2.5D physics backend.
    
    Uses the DynamicsEngine for kinematics simulation.
    "2.5D" means 2D plane with discrete altitude bands for aerial units.
    """
    
    def __init__(self):
        self.config: FullConfig = None
        self.dynamics: DynamicsEngine = None
        self.attackers: list[UnitState] = []
        self.defenders: list[UnitState] = []
    
    def initialize(self, config: FullConfig) -> None:
        """Initialize the backend with configuration."""
        self.config = config
        self.dynamics = DynamicsEngine(
            world_width=config.world.width,
            world_height=config.world.height,
            obstacles=config.world.obstacles,
            tick_rate=config.world.tick_rate,
            mobility_threshold=config.engagement.mobility_threshold,
            sensor_threshold=config.engagement.sensor_threshold,
        )
    
    def reset(self, attackers: list[UnitState], defenders: list[UnitState]) -> None:
        """Reset the simulation with initial unit states."""
        self.attackers = attackers
        self.defenders = defenders
    
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
        collisions = []
        
        # Step attackers
        for i, attacker in enumerate(self.attackers):
            if i < len(attacker_actions):
                action = attacker_actions[i]
            else:
                action = "NOOP"
            
            attacker, collision = self.dynamics.step_unit(attacker, action)
            self.attackers[i] = attacker
            collisions.append(collision)
        
        # Step defenders
        for i, defender in enumerate(self.defenders):
            if i < len(defender_actions):
                action = defender_actions[i]
            else:
                action = "NOOP"
            
            defender, collision = self.dynamics.step_unit(defender, action)
            self.defenders[i] = defender
            collisions.append(collision)
        
        return self.attackers, self.defenders, collisions
    
    def get_unit_states(self) -> tuple[list[UnitState], list[UnitState]]:
        """Get current unit states."""
        return self.attackers, self.defenders
    
    def close(self) -> None:
        """Clean up backend resources."""
        pass  # No resources to clean up
    
    def check_line_of_sight(
        self, x1: float, y1: float, x2: float, y2: float, alt1: int = 0, alt2: int = 0
    ) -> bool:
        """Check line of sight between two points."""
        return self.dynamics.check_line_of_sight(x1, y1, x2, y2, alt1, alt2)
