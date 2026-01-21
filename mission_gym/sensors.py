"""Sensor simulation for units."""

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from mission_gym.config import SensorConfig
from mission_gym.dynamics import UnitState, DynamicsEngine


@dataclass
class Detection:
    """Represents a sensor detection."""
    target_id: int
    target_team: str
    range: float
    bearing: float  # relative to unit heading
    x: float
    y: float
    detected_by: str  # sensor type


class SensorSystem:
    """Handles sensor simulation for all units."""
    
    def __init__(
        self,
        sensor_configs: dict[str, SensorConfig],
        dynamics: DynamicsEngine,
        sensor_threshold: float,
        rng: np.random.Generator,
    ):
        self.sensor_configs = sensor_configs
        self.dynamics = dynamics
        self.sensor_threshold = sensor_threshold
        self.rng = rng
    
    def get_detections(
        self,
        observer: UnitState,
        targets: list[UnitState],
    ) -> list[Detection]:
        """
        Get all detections from a unit's sensors.
        
        Args:
            observer: The observing unit
            targets: List of potential targets
        
        Returns:
            List of detections
        """
        if observer.is_disabled:
            return []
        
        detections = []
        config = observer.type_config
        if config is None:
            return []
        
        # Calculate sensor degradation
        range_mult = 1.0
        if observer.integrity <= self.sensor_threshold:
            range_mult = 0.5
        
        # Scan boost
        if observer.scan_active > 0:
            range_mult *= 1.5  # From scan config
        
        for sensor_name in config.sensors:
            sensor_cfg = self.sensor_configs.get(sensor_name)
            if sensor_cfg is None:
                continue
            
            effective_range = sensor_cfg.max_range * range_mult
            
            for target in targets:
                if target.is_disabled:
                    continue
                if target.unit_id == observer.unit_id:
                    continue
                
                detection = self._check_detection(
                    observer, target, sensor_cfg, effective_range
                )
                if detection is not None:
                    detections.append(detection)
        
        return detections
    
    def _check_detection(
        self,
        observer: UnitState,
        target: UnitState,
        sensor: SensorConfig,
        effective_range: float,
    ) -> Optional[Detection]:
        """Check if a target is detected by a sensor."""
        # Calculate range
        dist = observer.distance_to(target.x, target.y)
        if dist > effective_range:
            return None
        
        # Calculate relative bearing
        rel_bearing = observer.relative_bearing_to(target.x, target.y)
        
        # Check FOV
        half_fov = sensor.fov / 2
        if abs(rel_bearing) > half_fov:
            return None
        
        # Check LOS for camera sensors
        if sensor.requires_los:
            has_los = self.dynamics.check_line_of_sight(
                observer.x, observer.y, target.x, target.y,
                observer.altitude, target.altitude
            )
            if not has_los:
                return None
        
        # Apply noise for radar
        detected_range = dist
        detected_bearing = rel_bearing
        if sensor.type == "radar":
            if sensor.range_noise_std > 0:
                detected_range += self.rng.normal(0, sensor.range_noise_std)
            if sensor.bearing_noise_std > 0:
                detected_bearing += self.rng.normal(0, sensor.bearing_noise_std)
        
        # Calculate detected position
        absolute_bearing = observer.heading + detected_bearing
        detected_x = observer.x + detected_range * math.cos(math.radians(absolute_bearing))
        detected_y = observer.y + detected_range * math.sin(math.radians(absolute_bearing))
        
        return Detection(
            target_id=target.unit_id,
            target_team=target.team,
            range=detected_range,
            bearing=detected_bearing,
            x=detected_x,
            y=detected_y,
            detected_by=sensor.type,
        )
    
    def get_lidar_readings(
        self,
        observer: UnitState,
        all_units: list[UnitState],
    ) -> np.ndarray:
        """
        Get lidar distance readings for a unit.
        
        Returns array of distances for each ray.
        """
        if observer.is_disabled:
            return np.array([])
        
        config = observer.type_config
        if config is None:
            return np.array([])
        
        # Find lidar sensor
        lidar_cfg = None
        for sensor_name in config.sensors:
            sensor = self.sensor_configs.get(sensor_name)
            if sensor and sensor.type == "lidar":
                lidar_cfg = sensor
                break
        
        if lidar_cfg is None:
            return np.array([])
        
        # Calculate range multiplier
        range_mult = 1.0
        if observer.integrity <= self.sensor_threshold:
            range_mult = 0.5
        if observer.scan_active > 0:
            range_mult *= 1.5
        
        max_range = lidar_cfg.max_range * range_mult
        num_rays = lidar_cfg.num_rays
        half_fov = lidar_cfg.fov / 2
        
        readings = np.full(num_rays, max_range, dtype=np.float32)
        
        # Calculate ray angles
        if num_rays == 1:
            angles = [observer.heading]
        else:
            angles = [
                observer.heading - half_fov + (lidar_cfg.fov * i / (num_rays - 1))
                for i in range(num_rays)
            ]
        
        for i, angle in enumerate(angles):
            angle_rad = math.radians(angle)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            
            # Check obstacles
            min_dist = max_range
            if observer.altitude == 0:  # Only ground level lidar hits obstacles
                for obs in self.dynamics.obstacles:
                    dist = self._ray_obstacle_distance(
                        observer.x, observer.y, dx, dy, max_range, obs
                    )
                    if dist < min_dist:
                        min_dist = dist
            
            # Check other units
            for unit in all_units:
                if unit.unit_id == observer.unit_id:
                    continue
                if unit.is_disabled:
                    continue
                # Only detect units at same altitude (simplified)
                if abs(unit.altitude - observer.altitude) > 0:
                    continue
                
                dist = self._ray_circle_distance(
                    observer.x, observer.y, dx, dy, max_range,
                    unit.x, unit.y, unit.type_config.radius if unit.type_config else 0.5
                )
                if dist < min_dist:
                    min_dist = dist
            
            readings[i] = min_dist
        
        return readings
    
    def _ray_obstacle_distance(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float, obs
    ) -> float:
        """Calculate distance from ray origin to obstacle intersection."""
        if obs.type == "circle":
            return self._ray_circle_distance(
                ox, oy, dx, dy, max_range, obs.x, obs.y, obs.radius
            )
        elif obs.type == "rect":
            return self._ray_rect_distance(ox, oy, dx, dy, max_range, obs)
        return max_range
    
    def _ray_circle_distance(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float,
        cx: float, cy: float, r: float
    ) -> float:
        """Calculate distance from ray origin to circle intersection."""
        fx = ox - cx
        fy = oy - cy
        
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return max_range
        
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        if t1 >= 0:
            return min(t1, max_range)
        if t2 >= 0:
            return min(t2, max_range)
        return max_range
    
    def _ray_rect_distance(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float, rect
    ) -> float:
        """Calculate distance from ray origin to rectangle intersection."""
        corners = rect.get_corners()
        if not corners:
            return max_range
        
        min_dist = max_range
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            dist = self._ray_segment_distance(ox, oy, dx, dy, max_range, x1, y1, x2, y2)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def _ray_segment_distance(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float,
        x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Calculate distance from ray origin to line segment intersection."""
        # Ray: P = O + t*D
        # Segment: P = P1 + s*(P2-P1)
        sx = x2 - x1
        sy = y2 - y1
        
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return max_range  # Parallel
        
        t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
        s = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
        
        if t >= 0 and 0 <= s <= 1:
            return min(t, max_range)
        return max_range
