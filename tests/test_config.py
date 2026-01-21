"""Tests for configuration loading."""

import pytest
from pathlib import Path

from mission_gym.config import (
    load_yaml,
    FullConfig,
    WorldConfig,
    ScenarioConfig,
    EngagementConfig,
    RewardConfig,
)


class TestYAMLLoading:
    """Test YAML file loading."""

    def test_load_world_yaml(self):
        """Test loading world.yaml."""
        data = load_yaml("world.yaml")
        assert data is not None
        assert "arena" in data
        assert "physics" in data

    def test_load_scenario_yaml(self):
        """Test loading scenario.yaml."""
        data = load_yaml("scenario.yaml")
        assert data is not None
        assert "name" in data
        assert "objective" in data

    def test_load_units_attackers_yaml(self):
        """Test loading units_attackers.yaml."""
        data = load_yaml("units_attackers.yaml")
        assert data is not None
        assert len(data) > 0  # At least one unit type

    def test_load_units_defenders_yaml(self):
        """Test loading units_defenders.yaml."""
        data = load_yaml("units_defenders.yaml")
        assert data is not None
        assert len(data) > 0

    def test_load_sensors_yaml(self):
        """Test loading sensors.yaml."""
        data = load_yaml("sensors.yaml")
        assert data is not None

    def test_load_engagement_yaml(self):
        """Test loading engagement.yaml."""
        data = load_yaml("engagement.yaml")
        assert data is not None
        assert "tag_beam" in data

    def test_load_reward_yaml(self):
        """Test loading reward.yaml."""
        data = load_yaml("reward.yaml")
        assert data is not None


class TestFullConfig:
    """Test full configuration loading."""

    def test_load_full_config(self):
        """Test loading the complete configuration."""
        config = FullConfig.load()
        assert config is not None

    def test_world_config(self):
        """Test world configuration properties."""
        config = FullConfig.load()
        assert config.world is not None
        assert config.world.width > 0
        assert config.world.height > 0
        assert config.world.tick_rate > 0

    def test_scenario_config(self):
        """Test scenario configuration properties."""
        config = FullConfig.load()
        assert config.scenario is not None
        assert config.scenario.name is not None
        assert config.scenario.objective is not None

    def test_engagement_config(self):
        """Test engagement configuration properties."""
        config = FullConfig.load()
        assert config.engagement is not None
        assert config.engagement.tag_range > 0
        assert config.engagement.tag_cooldown >= 0

    def test_attacker_types(self):
        """Test attacker unit types are loaded."""
        config = FullConfig.load()
        assert config.attacker_types is not None
        assert len(config.attacker_types) > 0

    def test_defender_types(self):
        """Test defender unit types are loaded."""
        config = FullConfig.load()
        assert config.defender_types is not None
        assert len(config.defender_types) > 0


class TestObstacles:
    """Test obstacle configuration."""

    def test_obstacles_loaded(self):
        """Test that obstacles are loaded from world config."""
        config = FullConfig.load()
        assert config.world.obstacles is not None
        assert len(config.world.obstacles) > 0

    def test_obstacle_types(self):
        """Test obstacle type validation."""
        config = FullConfig.load()
        for obs in config.world.obstacles:
            assert obs.type in ["circle", "rect"]

    def test_circle_obstacles(self):
        """Test circle obstacles have radius."""
        config = FullConfig.load()
        circles = [o for o in config.world.obstacles if o.type == "circle"]
        for circle in circles:
            assert circle.radius is not None
            assert circle.radius > 0

    def test_rect_obstacles(self):
        """Test rectangle obstacles have size."""
        config = FullConfig.load()
        rects = [o for o in config.world.obstacles if o.type == "rect"]
        for rect in rects:
            assert rect.width is not None
            assert rect.height is not None
            assert rect.width > 0
            assert rect.height > 0
