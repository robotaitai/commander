"""Configuration loading and management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml
import math


def get_config_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent / "configs"


def load_yaml(filename: str, config_dir: Optional[Path] = None) -> dict:
    """Load a YAML config file from the configs directory."""
    if config_dir is None:
        config_dir = get_config_dir()
    config_path = config_dir / filename
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
    def from_yaml(cls, config_dir: Optional[Path] = None) -> "WorldConfig":
        data = load_yaml("world.yaml", config_dir)
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
    def from_yaml(cls, config_dir: Optional[Path] = None) -> "ScenarioConfig":
        data = load_yaml("scenario.yaml", config_dir)
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


def load_unit_types(filename: str, config_dir: Optional[Path] = None) -> dict[str, UnitTypeConfig]:
    """Load unit type configurations from YAML."""
    data = load_yaml(filename, config_dir)
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


def load_sensors(config_dir: Optional[Path] = None) -> dict[str, SensorConfig]:
    """Load sensor configurations from YAML."""
    data = load_yaml("sensors.yaml", config_dir)
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
    # Enable/disable flags for checkpoint compatibility
    tag_enabled: bool
    scan_enabled: bool
    # Tag beam configuration
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
    def from_yaml(cls, config_dir: Optional[Path] = None) -> "EngagementConfig":
        data = load_yaml("engagement.yaml", config_dir)
        enable = data.get("enable", {})
        tag = data["tag_beam"]
        deg = tag["degradation"]
        scan = data["scan"]
        return cls(
            tag_enabled=enable.get("tag", False),
            scan_enabled=enable.get("scan", False),
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
    
    # Outcome penalties (terminal rewards)
    outcome_penalty_stalled: float
    outcome_penalty_timeout: float
    outcome_penalty_all_disabled: float
    
    # Toggles
    enable_detected_penalty: bool
    
    @classmethod
    def from_yaml(cls, config_dir: Optional[Path] = None) -> "RewardConfig":
        data = load_yaml("reward.yaml", config_dir)
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
            
            # Outcome penalties
            outcome_penalty_stalled=data.get("outcome_penalties", {}).get("stalled", -50.0),
            outcome_penalty_timeout=data.get("outcome_penalties", {}).get("timeout", -20.0),
            outcome_penalty_all_disabled=data.get("outcome_penalties", {}).get("all_disabled", -100.0),
            
            # Toggles
            enable_detected_penalty=enable.get("detected_time_penalty", True),
        )


@dataclass
class TerminationConfig:
    """Early termination configuration."""
    stagnation_seconds: float
    min_dist_epsilon: float
    early_success_capture_progress: Optional[float]
    
    @classmethod
    def from_yaml(cls, config_dir: Optional[Path] = None) -> "TerminationConfig":
        data = load_yaml("world.yaml", config_dir)
        term = data.get("termination", {})
        return cls(
            stagnation_seconds=term.get("stagnation_seconds", 30.0),
            min_dist_epsilon=term.get("min_dist_epsilon", 1.0),
            early_success_capture_progress=term.get("early_success_capture_progress"),
        )


@dataclass
class DefenderRandomizationConfig:
    """Defender domain randomization configuration."""
    mode_probs: dict[str, float]
    delay_min: float
    delay_max: float
    p_random_action: float
    aim_std: float
    patrol_jitter_enabled: bool
    patrol_jitter_radius: float
    
    @classmethod
    def from_yaml(cls, config_dir: Optional[Path] = None) -> "DefenderRandomizationConfig":
        try:
            data = load_yaml("defender_randomization.yaml", config_dir)
        except FileNotFoundError:
            # Return defaults if file doesn't exist
            return cls(
                mode_probs={"patrol": 0.5, "guard_objective": 0.3, "intercept": 0.2},
                delay_min=0.0,
                delay_max=0.5,
                p_random_action=0.05,
                aim_std=5.0,
                patrol_jitter_enabled=True,
                patrol_jitter_radius=10.0,
            )
        
        return cls(
            mode_probs=data.get("behavior_modes", {"patrol": 0.5, "guard_objective": 0.3, "intercept": 0.2}),
            delay_min=data.get("reaction", {}).get("delay_min", 0.0),
            delay_max=data.get("reaction", {}).get("delay_max", 0.5),
            p_random_action=data.get("reaction", {}).get("p_random_action", 0.05),
            aim_std=data.get("jitter", {}).get("aim_std", 5.0),
            patrol_jitter_enabled=data.get("patrol_randomization", {}).get("enabled", True),
            patrol_jitter_radius=data.get("patrol_randomization", {}).get("jitter_radius", 10.0),
        )


@dataclass
class ScenarioRandomizationConfig:
    """Scenario randomization configuration."""
    spawn_enabled: bool
    attacker_jitter_x: float
    attacker_jitter_y: float
    defender_jitter_x: float
    defender_jitter_y: float
    objective_enabled: bool
    objective_jitter_x: float
    objective_jitter_y: float
    
    @classmethod
    def from_yaml(cls, config_dir: Optional[Path] = None) -> "ScenarioRandomizationConfig":
        try:
            data = load_yaml("scenario_randomization.yaml", config_dir)
        except FileNotFoundError:
            # Return defaults (no randomization) if file doesn't exist
            return cls(
                spawn_enabled=False,
                attacker_jitter_x=0.0,
                attacker_jitter_y=0.0,
                defender_jitter_x=0.0,
                defender_jitter_y=0.0,
                objective_enabled=False,
                objective_jitter_x=0.0,
                objective_jitter_y=0.0,
            )
        
        spawn = data.get("spawn_randomization", {})
        objective = data.get("objective_randomization", {})
        
        return cls(
            spawn_enabled=spawn.get("enabled", False),
            attacker_jitter_x=spawn.get("attackers", {}).get("jitter_x", 0.0),
            attacker_jitter_y=spawn.get("attackers", {}).get("jitter_y", 0.0),
            defender_jitter_x=spawn.get("defenders", {}).get("jitter_x", 0.0),
            defender_jitter_y=spawn.get("defenders", {}).get("jitter_y", 0.0),
            objective_enabled=objective.get("enabled", False),
            objective_jitter_x=objective.get("jitter", {}).get("x_range", 0.0),
            objective_jitter_y=objective.get("jitter", {}).get("y_range", 0.0),
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
    termination: TerminationConfig
    defender_randomization: DefenderRandomizationConfig
    scenario_randomization: ScenarioRandomizationConfig
    
    @classmethod
    def load(cls, config_dir: Optional[Path] = None) -> "FullConfig":
        return cls(
            world=WorldConfig.from_yaml(config_dir),
            scenario=ScenarioConfig.from_yaml(config_dir),
            attacker_types=load_unit_types("units_attackers.yaml", config_dir),
            defender_types=load_unit_types("units_defenders.yaml", config_dir),
            sensors=load_sensors(config_dir),
            engagement=EngagementConfig.from_yaml(config_dir),
            reward=RewardConfig.from_yaml(config_dir),
            termination=TerminationConfig.from_yaml(config_dir),
            defender_randomization=DefenderRandomizationConfig.from_yaml(config_dir),
            scenario_randomization=ScenarioRandomizationConfig.from_yaml(config_dir),
        )
