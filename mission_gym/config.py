"""Configuration loading and management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
import math


def get_config_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent / "configs"


def load_yaml(filename: str) -> dict:
    """Load a YAML config file from the configs directory."""
    config_path = get_config_dir() / filename
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class Obstacle:
    """Represents an obstacle in the world."""
    type: str  # 'circle' or 'rect'
    x: float
    y: float
    # Circle attributes
    radius: float = 0.0
    # Rectangle attributes
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0  # degrees
    
    def get_corners(self) -> list[tuple[float, float]]:
        """Get corner points for rectangular obstacles."""
        if self.type != 'rect':
            return []
        
        # Half dimensions
        hw, hh = self.width / 2, self.height / 2
        # Local corners
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        
        # Rotate and translate
        angle_rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            rotated.append((rx, ry))
        return rotated


@dataclass
class WorldConfig:
    """World configuration."""
    width: float
    height: float
    obstacles: list[Obstacle]
    tick_rate: float
    action_repeat: int
    max_duration: float
    
    @classmethod
    def from_yaml(cls) -> "WorldConfig":
        data = load_yaml("world.yaml")
        obstacles = []
        for obs_data in data.get("obstacles", []):
            obstacles.append(Obstacle(
                type=obs_data["type"],
                x=obs_data["x"],
                y=obs_data["y"],
                radius=obs_data.get("radius", 0.0),
                width=obs_data.get("width", 0.0),
                height=obs_data.get("height", 0.0),
                angle=obs_data.get("angle", 0.0),
            ))
        return cls(
            width=data["arena"]["width"],
            height=data["arena"]["height"],
            obstacles=obstacles,
            tick_rate=data["physics"]["tick_rate"],
            action_repeat=data["physics"]["action_repeat"],
            max_duration=data["episode"]["max_duration"],
        )


@dataclass
class ObjectiveConfig:
    """Objective zone configuration."""
    type: str
    x: float
    y: float
    radius: float
    capture_time_required: float


@dataclass
class UnitSpawn:
    """Unit spawn configuration."""
    unit_type: str
    x: float
    y: float
    heading: float
    altitude: int = 0
    patrol_waypoints: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class ScenarioConfig:
    """Scenario configuration."""
    name: str
    objective: ObjectiveConfig
    attackers: list[UnitSpawn]
    defenders: list[UnitSpawn]
    
    @classmethod
    def from_yaml(cls) -> "ScenarioConfig":
        data = load_yaml("scenario.yaml")
        obj_data = data["objective"]
        objective = ObjectiveConfig(
            type=obj_data["type"],
            x=obj_data["x"],
            y=obj_data["y"],
            radius=obj_data["radius"],
            capture_time_required=obj_data["capture_time_required"],
        )
        
        attackers = []
        for atk in data.get("attackers", []):
            spawn = atk["spawn"]
            attackers.append(UnitSpawn(
                unit_type=atk["unit_type"],
                x=spawn["x"],
                y=spawn["y"],
                heading=spawn["heading"],
                altitude=spawn.get("altitude", 0),
            ))
        
        defenders = []
        for dfn in data.get("defenders", []):
            spawn = dfn["spawn"]
            waypoints = dfn.get("patrol_waypoints", [])
            defenders.append(UnitSpawn(
                unit_type=dfn["unit_type"],
                x=spawn["x"],
                y=spawn["y"],
                heading=spawn["heading"],
                altitude=spawn.get("altitude", 0),
                patrol_waypoints=[tuple(wp) for wp in waypoints],
            ))
        
        return cls(
            name=data["name"],
            objective=objective,
            attackers=attackers,
            defenders=defenders,
        )


@dataclass
class UnitTypeConfig:
    """Unit type configuration."""
    name: str
    category: str  # 'ground' or 'air'
    max_speed: float
    max_accel: float
    max_turn_rate: float
    radius: float
    initial_integrity: float
    sensors: list[str]
    actions: list[str]
    altitude_bands: int = 1
    altitude_change_time: float = 1.0
    behavior: dict = field(default_factory=dict)
    initial_speed: float = 0.0  # Initial speed when spawned (0 = stationary)


def load_unit_types(filename: str) -> dict[str, UnitTypeConfig]:
    """Load unit type configurations from YAML."""
    data = load_yaml(filename)
    unit_types = {}
    for name, cfg in data.get("unit_types", {}).items():
        unit_types[name] = UnitTypeConfig(
            name=name,
            category=cfg["category"],
            max_speed=cfg["max_speed"],
            max_accel=cfg["max_accel"],
            max_turn_rate=cfg["max_turn_rate"],
            radius=cfg["radius"],
            initial_integrity=cfg["initial_integrity"],
            sensors=cfg.get("sensors", []),
            actions=cfg.get("actions", []),
            altitude_bands=cfg.get("altitude_bands", 1),
            altitude_change_time=cfg.get("altitude_change_time", 1.0),
            behavior=cfg.get("behavior", {}),
            initial_speed=cfg.get("initial_speed", 0.0),
        )
    return unit_types


@dataclass
class SensorConfig:
    """Sensor configuration."""
    name: str
    type: str  # 'lidar', 'radar', 'camera'
    max_range: float
    fov: float
    update_rate: float
    # Lidar specific
    num_rays: int = 0
    # Radar specific
    range_noise_std: float = 0.0
    bearing_noise_std: float = 0.0
    # Camera specific
    requires_los: bool = True


def load_sensors() -> dict[str, SensorConfig]:
    """Load sensor configurations from YAML."""
    data = load_yaml("sensors.yaml")
    sensors = {}
    for name, cfg in data.get("sensors", {}).items():
        sensors[name] = SensorConfig(
            name=name,
            type=cfg["type"],
            max_range=cfg["max_range"],
            fov=cfg["fov"],
            update_rate=cfg["update_rate"],
            num_rays=cfg.get("num_rays", 0),
            range_noise_std=cfg.get("range_noise_std", 0.0),
            bearing_noise_std=cfg.get("bearing_noise_std", 0.0),
            requires_los=cfg.get("requires_los", True),
        )
    return sensors


@dataclass
class EngagementConfig:
    """Engagement (tag) configuration."""
    tag_range: float
    tag_optimal_range: float
    tag_fov: float
    tag_requires_los: bool
    tag_cooldown: float
    tag_damage: float
    tag_min_damage: float
    mobility_threshold: float
    sensor_threshold: float
    disabled_threshold: float
    scan_duration: float
    scan_range_boost: float
    scan_cooldown: float
    
    @classmethod
    def from_yaml(cls) -> "EngagementConfig":
        data = load_yaml("engagement.yaml")
        tag = data["tag_beam"]
        deg = tag["degradation"]
        scan = data["scan"]
        return cls(
            tag_range=tag["range"],
            tag_optimal_range=tag.get("optimal_range", tag["range"] * 0.5),
            tag_fov=tag["fov"],
            tag_requires_los=tag["requires_los"],
            tag_cooldown=tag["cooldown"],
            tag_damage=tag["damage"],
            tag_min_damage=tag.get("min_damage", tag["damage"] * 0.4),
            mobility_threshold=deg["mobility_threshold"],
            sensor_threshold=deg["sensor_threshold"],
            disabled_threshold=deg["disabled_threshold"],
            scan_duration=scan["duration"],
            scan_range_boost=scan["range_boost"],
            scan_cooldown=scan["cooldown"],
        )


@dataclass
class RewardConfig:
    """Reward function configuration."""
    # Objective rewards
    capture_progress: float
    win_bonus: float
    zone_entry_bonus: float
    zone_time: float
    
    # Shaping rewards
    min_dist_potential: float
    ring_bonus: float
    ring_distances: list  # List of distance milestones
    approach_objective: float
    spread_formation: float
    
    # Engagement bonuses
    tag_hit_bonus: float
    defender_disabled_bonus: float
    
    # Penalties
    time_penalty: float
    integrity_loss_penalty: float
    collision_penalty: float
    unit_disabled_penalty: float
    detected_time_penalty: float
    
    # Toggles
    enable_detected_penalty: bool
    
    @classmethod
    def from_yaml(cls) -> "RewardConfig":
        data = load_yaml("reward.yaml")
        weights = data.get("weights", {})
        enable = data.get("enable", {})
        shaping = data.get("shaping", {})
        
        return cls(
            # Objective rewards
            capture_progress=weights.get("capture_progress", 2.0),
            win_bonus=weights.get("win_bonus", 200.0),
            zone_entry_bonus=weights.get("zone_entry_bonus", 20.0),
            zone_time=weights.get("zone_time", 2.0),
            
            # Shaping rewards
            min_dist_potential=weights.get("min_dist_potential", 0.5),
            ring_bonus=weights.get("ring_bonus", 5.0),
            ring_distances=data.get("ring_distances", [80, 60, 40, 25, 15]),
            approach_objective=weights.get("approach_objective", 0.0),  # Disabled by default
            spread_formation=weights.get("spread_formation", 0.005),
            
            # Engagement bonuses
            tag_hit_bonus=weights.get("tag_hit_bonus", 0.2),
            defender_disabled_bonus=weights.get("defender_disabled_bonus", 10.0),
            
            # Penalties
            time_penalty=weights.get("time_penalty", -0.001),
            integrity_loss_penalty=weights.get("integrity_loss_penalty", -0.1),
            collision_penalty=weights.get("collision_penalty", -0.5),
            unit_disabled_penalty=weights.get("unit_disabled_penalty", -20.0),
            detected_time_penalty=weights.get("detected_time_penalty", -0.05),
            
            # Toggles
            enable_detected_penalty=enable.get("detected_time_penalty", True),
        )


@dataclass
class FullConfig:
    """Complete configuration for the environment."""
    world: WorldConfig
    scenario: ScenarioConfig
    attacker_types: dict[str, UnitTypeConfig]
    defender_types: dict[str, UnitTypeConfig]
    sensors: dict[str, SensorConfig]
    engagement: EngagementConfig
    reward: RewardConfig
    
    @classmethod
    def load(cls) -> "FullConfig":
        return cls(
            world=WorldConfig.from_yaml(),
            scenario=ScenarioConfig.from_yaml(),
            attacker_types=load_unit_types("units_attackers.yaml"),
            defender_types=load_unit_types("units_defenders.yaml"),
            sensors=load_sensors(),
            engagement=EngagementConfig.from_yaml(),
            reward=RewardConfig.from_yaml(),
        )
