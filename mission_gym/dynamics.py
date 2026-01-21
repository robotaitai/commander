"""Unit dynamics and physics simulation."""

from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np

from mission_gym.config import UnitTypeConfig, Obstacle


# Action constants for ground units (UGV)
UGV_ACTIONS = [
    "NOOP", "THROTTLE_UP", "THROTTLE_DOWN", "TURN_LEFT", "TURN_RIGHT",
    "BRAKE", "HOLD", "TAG", "SCAN"
]

# Action constants for air units (UAV)
UAV_ACTIONS = [
    "NOOP", "THROTTLE_UP", "THROTTLE_DOWN", "YAW_LEFT", "YAW_RIGHT",
    "ALT_UP", "ALT_DOWN", "HOLD", "TAG", "SCAN"
]


def get_action_list(category: str) -> list[str]:
    """Get the action list for a unit category."""
    if category == "ground":
        return UGV_ACTIONS
    else:
        return UAV_ACTIONS


def get_num_actions(category: str) -> int:
    """Get the number of actions for a unit category."""
    return len(get_action_list(category))


@dataclass
class UnitState:
    """Represents the dynamic state of a unit."""
    # Position and orientation
    x: float
    y: float
    heading: float  # degrees, 0 = east, CCW positive
    
    # Velocity
    speed: float = 0.0
    
    # For UAV: altitude band (0, 1, 2)
    altitude: int = 0
    altitude_transition: float = 0.0  # progress toward next altitude
    target_altitude: int = 0
    
    # Status
    integrity: float = 100.0
    is_disabled: bool = False
    
    # Cooldowns (seconds remaining)
    tag_cooldown: float = 0.0
    scan_cooldown: float = 0.0
    scan_active: float = 0.0  # remaining scan duration
    
    # Unit info (set during initialization)
    unit_id: int = 0
    unit_type: str = ""
    category: str = "ground"
    team: str = "attacker"  # 'attacker' or 'defender'
    
    # Type config reference (set during initialization)
    type_config: Optional[UnitTypeConfig] = None
    
    def get_heading_rad(self) -> float:
        """Get heading in radians."""
        return math.radians(self.heading)
    
    def get_forward_vector(self) -> tuple[float, float]:
        """Get the unit's forward direction as (dx, dy)."""
        rad = self.get_heading_rad()
        return (math.cos(rad), math.sin(rad))
    
    def distance_to(self, other_x: float, other_y: float) -> float:
        """Calculate distance to a point."""
        return math.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)
    
    def bearing_to(self, other_x: float, other_y: float) -> float:
        """Calculate bearing to a point in degrees."""
        dx = other_x - self.x
        dy = other_y - self.y
        return math.degrees(math.atan2(dy, dx))
    
    def relative_bearing_to(self, other_x: float, other_y: float) -> float:
        """Calculate relative bearing (angle from heading) to a point."""
        absolute_bearing = self.bearing_to(other_x, other_y)
        relative = absolute_bearing - self.heading
        # Normalize to [-180, 180]
        while relative > 180:
            relative -= 360
        while relative < -180:
            relative += 360
        return relative
    
    def apply_mobility_degradation(self) -> float:
        """Get speed multiplier based on integrity."""
        if self.type_config is None:
            return 1.0
        # This is checked against engagement config thresholds
        return 1.0  # Actual degradation applied in dynamics step
    
    def to_vector(self) -> np.ndarray:
        """Convert state to a feature vector for observations."""
        return np.array([
            self.x,
            self.y,
            self.heading / 180.0,  # Normalize to [-1, 1]
            self.speed,
            self.integrity / 100.0,  # Normalize to [0, 1]
            self.tag_cooldown,
            self.scan_cooldown,
            float(self.altitude),
            float(self.is_disabled),
        ], dtype=np.float32)


class DynamicsEngine:
    """Handles physics simulation for all units."""
    
    def __init__(
        self,
        world_width: float,
        world_height: float,
        obstacles: list[Obstacle],
        tick_rate: float,
        mobility_threshold: float,
        sensor_threshold: float,
    ):
        self.world_width = world_width
        self.world_height = world_height
        self.obstacles = obstacles
        self.dt = 1.0 / tick_rate
        self.mobility_threshold = mobility_threshold
        self.sensor_threshold = sensor_threshold
    
    def step_unit(self, state: UnitState, action: str) -> tuple[UnitState, bool]:
        """
        Step a single unit forward by one tick.
        
        Returns:
            Updated state and collision flag.
        """
        if state.is_disabled:
            return state, False
        
        config = state.type_config
        if config is None:
            return state, False
        
        # Update cooldowns
        state.tag_cooldown = max(0.0, state.tag_cooldown - self.dt)
        state.scan_cooldown = max(0.0, state.scan_cooldown - self.dt)
        state.scan_active = max(0.0, state.scan_active - self.dt)
        
        # Calculate mobility multiplier based on integrity
        mobility_mult = 1.0
        if state.integrity <= self.mobility_threshold:
            mobility_mult = 0.5
        
        # Calculate effective limits
        max_speed = config.max_speed * mobility_mult
        max_accel = config.max_accel * mobility_mult
        max_turn = config.max_turn_rate * mobility_mult
        
        # Process action
        target_speed = state.speed
        turn_rate = 0.0
        altitude_change = 0
        
        if action == "NOOP":
            pass
        elif action == "THROTTLE_UP":
            target_speed = min(state.speed + max_accel * self.dt, max_speed)
        elif action == "THROTTLE_DOWN":
            target_speed = max(state.speed - max_accel * self.dt, 0.0)
        elif action in ("TURN_LEFT", "YAW_LEFT"):
            turn_rate = max_turn
        elif action in ("TURN_RIGHT", "YAW_RIGHT"):
            turn_rate = -max_turn
        elif action == "BRAKE":
            target_speed = max(state.speed - max_accel * 2 * self.dt, 0.0)
        elif action == "HOLD":
            target_speed = 0.0
        elif action == "ALT_UP":
            if state.category == "air" and state.altitude < config.altitude_bands - 1:
                altitude_change = 1
        elif action == "ALT_DOWN":
            if state.category == "air" and state.altitude > 0:
                altitude_change = -1
        # TAG and SCAN are handled by engagement system
        
        # Update speed
        state.speed = target_speed
        
        # Update heading
        state.heading += turn_rate * self.dt
        # Normalize heading to [0, 360)
        state.heading = state.heading % 360
        
        # Update altitude for UAV
        if state.category == "air" and altitude_change != 0:
            state.target_altitude = state.altitude + altitude_change
            state.target_altitude = max(0, min(state.target_altitude, config.altitude_bands - 1))
        
        # Process altitude transition
        if state.category == "air" and state.altitude != state.target_altitude:
            state.altitude_transition += self.dt / config.altitude_change_time
            if state.altitude_transition >= 1.0:
                state.altitude = state.target_altitude
                state.altitude_transition = 0.0
        
        # Calculate new position
        dx, dy = state.get_forward_vector()
        new_x = state.x + dx * state.speed * self.dt
        new_y = state.y + dy * state.speed * self.dt
        
        # Check collision with obstacles (only for ground units or UAV at altitude 0)
        collision = False
        if state.category == "ground" or state.altitude == 0:
            collision = self._check_collision(new_x, new_y, config.radius)
            if collision:
                # Stop at current position
                new_x, new_y = state.x, state.y
                state.speed = 0.0
        
        # Clamp to world bounds
        new_x = max(config.radius, min(self.world_width - config.radius, new_x))
        new_y = max(config.radius, min(self.world_height - config.radius, new_y))
        
        state.x = new_x
        state.y = new_y
        
        return state, collision
    
    def _check_collision(self, x: float, y: float, radius: float) -> bool:
        """Check if a circular unit collides with any obstacle."""
        for obs in self.obstacles:
            if obs.type == "circle":
                dist = math.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2)
                if dist < radius + obs.radius:
                    return True
            elif obs.type == "rect":
                if self._circle_rect_collision(x, y, radius, obs):
                    return True
        return False
    
    def _circle_rect_collision(self, cx: float, cy: float, r: float, rect: Obstacle) -> bool:
        """Check collision between circle and rotated rectangle."""
        # Transform circle center to rectangle's local space
        angle_rad = math.radians(-rect.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        # Translate
        dx = cx - rect.x
        dy = cy - rect.y
        
        # Rotate
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        # Find closest point on rectangle
        hw, hh = rect.width / 2, rect.height / 2
        closest_x = max(-hw, min(hw, local_x))
        closest_y = max(-hh, min(hh, local_y))
        
        # Check distance
        dist_sq = (local_x - closest_x) ** 2 + (local_y - closest_y) ** 2
        return dist_sq < r * r
    
    def check_line_of_sight(
        self, x1: float, y1: float, x2: float, y2: float, alt1: int = 0, alt2: int = 0
    ) -> bool:
        """
        Check if there's line of sight between two points.
        
        LOS is blocked by obstacles only if both points are at altitude 0.
        """
        if alt1 > 0 or alt2 > 0:
            return True  # Air units can see over obstacles
        
        for obs in self.obstacles:
            if obs.type == "circle":
                if self._line_circle_intersection(x1, y1, x2, y2, obs.x, obs.y, obs.radius):
                    return False
            elif obs.type == "rect":
                if self._line_rect_intersection(x1, y1, x2, y2, obs):
                    return False
        return True
    
    def _line_circle_intersection(
        self, x1: float, y1: float, x2: float, y2: float,
        cx: float, cy: float, r: float
    ) -> bool:
        """Check if line segment intersects circle."""
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy
        
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False
        
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)
    
    def _line_rect_intersection(self, x1: float, y1: float, x2: float, y2: float, rect: Obstacle) -> bool:
        """Check if line segment intersects rotated rectangle."""
        corners = rect.get_corners()
        if not corners:
            return False
        
        # Check intersection with each edge
        for i in range(4):
            rx1, ry1 = corners[i]
            rx2, ry2 = corners[(i + 1) % 4]
            if self._segments_intersect(x1, y1, x2, y2, rx1, ry1, rx2, ry2):
                return True
        
        # Check if line is completely inside rectangle
        if self._point_in_polygon(x1, y1, corners):
            return True
        
        return False
    
    def _segments_intersect(
        self, x1: float, y1: float, x2: float, y2: float,
        x3: float, y3: float, x4: float, y4: float
    ) -> bool:
        """Check if two line segments intersect."""
        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
        
        return (
            ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
            ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)
        )
    
    def _point_in_polygon(self, x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
