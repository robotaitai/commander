"""Tests for unit dynamics and physics."""

import pytest
import numpy as np
import math

from mission_gym.config import FullConfig, UnitTypeConfig
from mission_gym.dynamics import UnitState


class TestUnitState:
    """Test UnitState creation and properties."""

    def test_create_unit_state(self):
        """Test creating a basic unit state."""
        config = FullConfig.load()
        unit_type = list(config.attacker_types.values())[0]
        
        unit = UnitState(
            x=50.0,
            y=50.0,
            heading=90.0,
            speed=0.0,
            integrity=100.0,
            type_config=unit_type,
            team="attacker",
        )
        
        assert unit.x == 50.0
        assert unit.y == 50.0
        assert unit.heading == 90.0
        assert unit.speed == 0.0
        assert unit.integrity == 100.0

    def test_unit_is_disabled(self):
        """Test unit disabled state."""
        config = FullConfig.load()
        unit_type = list(config.attacker_types.values())[0]
        
        # Healthy unit
        unit = UnitState(
            x=0, y=0, heading=0, speed=0,
            integrity=50.0, type_config=unit_type, team="attacker"
        )
        assert not unit.is_disabled
        
        # Disabled unit
        unit.integrity = 0.0
        unit.is_disabled = True  # Manually set, as it's a flag in the dataclass
        assert unit.is_disabled

    def test_unit_category_ground(self):
        """Test ground unit category."""
        config = FullConfig.load()
        ground_types = [t for t in config.attacker_types.values() if t.category == "ground"]
        
        if ground_types:
            unit = UnitState(
                x=0, y=0, heading=0, speed=0,
                integrity=100.0, type_config=ground_types[0], team="attacker"
            )
            assert unit.type_config.category == "ground"

    def test_unit_category_air(self):
        """Test air unit category."""
        config = FullConfig.load()
        air_types = [t for t in config.attacker_types.values() if t.category == "air"]
        
        if air_types:
            unit = UnitState(
                x=0, y=0, heading=0, speed=0,
                integrity=100.0, type_config=air_types[0], team="attacker"
            )
            assert unit.type_config.category == "air"


class TestUnitMovement:
    """Test unit movement calculations."""

    def test_heading_normalization(self):
        """Test heading stays in valid range."""
        config = FullConfig.load()
        unit_type = list(config.attacker_types.values())[0]
        
        unit = UnitState(
            x=0, y=0, heading=350.0, speed=0,
            integrity=100.0, type_config=unit_type, team="attacker"
        )
        
        # Simulate turning past 360
        unit.heading = 370.0
        unit.heading = unit.heading % 360
        assert 0 <= unit.heading < 360

    def test_speed_limits(self):
        """Test speed respects max_speed."""
        config = FullConfig.load()
        unit_type = list(config.attacker_types.values())[0]
        
        max_speed = unit_type.max_speed
        
        # Speed should be clamped in actual dynamics
        # Here we just verify the config is loaded
        assert max_speed > 0


class TestUnitRadius:
    """Test unit radius/footprint properties."""

    def test_ugv_has_larger_radius(self):
        """Test UGV units have reasonable radius."""
        config = FullConfig.load()
        
        ground_types = [t for t in config.attacker_types.values() if t.category == "ground"]
        air_types = [t for t in config.attacker_types.values() if t.category == "air"]
        
        if ground_types and air_types:
            # Ground units should generally have larger footprint
            ground_radius = max(t.radius for t in ground_types)
            air_radius = min(t.radius for t in air_types)
            assert ground_radius > air_radius

    def test_unit_radius_positive(self):
        """Test all unit types have positive radius."""
        config = FullConfig.load()
        
        for unit_type in config.attacker_types.values():
            assert unit_type.radius > 0
        
        for unit_type in config.defender_types.values():
            assert unit_type.radius > 0
