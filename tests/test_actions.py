"""Tests for high-level action space."""

import pytest
import numpy as np
import math

from mission_gym.env import MissionGymEnv
from mission_gym.dynamics import (
    HIGH_LEVEL_ACTIONS,
    ACTION_TO_HEADING,
    UnitState,
    DynamicsEngine,
)
from mission_gym.config import UnitTypeConfig, Obstacle


class TestHighLevelActions:
    """Test high-level action definitions and mapping."""
    
    def test_action_list_length(self):
        """Verify we have exactly 9 actions."""
        assert len(HIGH_LEVEL_ACTIONS) == 9
    
    def test_action_names(self):
        """Verify action names are correct."""
        expected = [
            "STOP", "NORTH", "NORTHEAST", "EAST", "SOUTHEAST",
            "SOUTH", "SOUTHWEST", "WEST", "NORTHWEST"
        ]
        assert HIGH_LEVEL_ACTIONS == expected
    
    def test_action_to_heading_mapping(self):
        """Verify action index to heading mapping."""
        assert ACTION_TO_HEADING[0] is None  # STOP
        assert ACTION_TO_HEADING[1] == 90.0  # NORTH
        assert ACTION_TO_HEADING[2] == 45.0  # NORTHEAST
        assert ACTION_TO_HEADING[3] == 0.0   # EAST
        assert ACTION_TO_HEADING[4] == 315.0 # SOUTHEAST
        assert ACTION_TO_HEADING[5] == 270.0 # SOUTH
        assert ACTION_TO_HEADING[6] == 225.0 # SOUTHWEST
        assert ACTION_TO_HEADING[7] == 180.0 # WEST
        assert ACTION_TO_HEADING[8] == 135.0 # NORTHWEST
    
    def test_compass_directions_coverage(self):
        """Verify we cover all 8 compass directions."""
        headings = [h for h in ACTION_TO_HEADING.values() if h is not None]
        expected_headings = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        assert sorted(headings) == sorted(expected_headings)


class TestEnvironmentActionSpace:
    """Test action space in the environment."""
    
    def test_action_space_shape(self):
        """Verify action space has correct shape."""
        env = MissionGymEnv()
        assert env.action_space.shape == (4,)  # 4 units
        assert all(env.action_space.nvec == 9)  # Each unit has 9 actions
    
    def test_action_space_sample(self):
        """Verify sampled actions are valid."""
        env = MissionGymEnv()
        for _ in range(10):
            action = env.action_space.sample()
            assert action.shape == (4,)
            assert all(0 <= a <= 8 for a in action)
        env.close()
    
    def test_step_with_valid_actions(self):
        """Test stepping environment with all valid actions."""
        env = MissionGymEnv()
        obs, info = env.reset()
        
        # Test each action
        for action_idx in range(9):
            action = np.array([action_idx, 0, 0, 0], dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
        
        env.close()
    
    def test_all_units_different_actions(self):
        """Test all units executing different actions simultaneously."""
        env = MissionGymEnv()
        obs, info = env.reset()
        
        # Each unit does a different action
        action = np.array([1, 3, 5, 7], dtype=np.int32)  # N, E, S, W
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        env.close()


class TestDynamicsWithHighLevelActions:
    """Test dynamics engine with high-level actions."""
    
    def create_test_unit(self) -> tuple[UnitState, UnitTypeConfig]:
        """Create a test unit and config."""
        config = UnitTypeConfig(
            name="TEST_UGV",
            category="ground",
            max_speed=10.0,
            max_accel=5.0,
            max_turn_rate=90.0,
            radius=2.0,
            initial_integrity=100.0,
            initial_speed=0.0,
            sensors=[],
            actions=HIGH_LEVEL_ACTIONS,
        )
        
        state = UnitState(
            x=50.0,
            y=50.0,
            heading=0.0,
            speed=0.0,
            type_config=config,
        )
        
        return state, config
    
    def test_stop_action(self):
        """Test STOP action sets target speed to 0."""
        state, config = self.create_test_unit()
        state.speed = 5.0  # Start with some speed
        
        engine = DynamicsEngine(
            world_width=200.0,
            world_height=200.0,
            obstacles=[],
            tick_rate=20.0,
            mobility_threshold=30.0,
            sensor_threshold=50.0,
        )
        
        # Execute STOP action
        for _ in range(20):  # Multiple steps
            state, collision = engine.step_unit(state, "STOP")
        
        # Speed should be 0 or very close
        assert state.speed < 0.5
    
    def test_directional_action_sets_target_heading(self):
        """Test directional actions set correct target heading."""
        state, config = self.create_test_unit()
        
        engine = DynamicsEngine(
            world_width=200.0,
            world_height=200.0,
            obstacles=[],
            tick_rate=20.0,
            mobility_threshold=30.0,
            sensor_threshold=50.0,
        )
        
        # Execute NORTH action
        for _ in range(20):
            state, collision = engine.step_unit(state, "NORTH")
        
        # Heading should be turning toward 90°
        assert state.target_heading == 90.0
        # Speed should be increasing
        assert state.speed > 0.0
    
    def test_smooth_heading_convergence(self):
        """Test unit smoothly converges to target heading."""
        state, config = self.create_test_unit()
        state.heading = 0.0  # Start facing EAST
        
        engine = DynamicsEngine(
            world_width=200.0,
            world_height=200.0,
            obstacles=[],
            tick_rate=20.0,
            mobility_threshold=30.0,
            sensor_threshold=50.0,
        )
        
        # Execute NORTH action (target 90°)
        previous_heading = 0.0
        for _ in range(40):
            state, collision = engine.step_unit(state, "NORTH")
            # Heading should gradually increase toward 90°
            if state.heading < 90.0:
                assert state.heading >= previous_heading or abs(state.heading - previous_heading) > 180
            previous_heading = state.heading
        
        # Should be close to 90° after 2 seconds (40 steps at 20Hz)
        assert 85.0 <= state.heading <= 95.0
    
    def test_smooth_speed_convergence(self):
        """Test unit smoothly accelerates to target speed."""
        state, config = self.create_test_unit()
        state.speed = 0.0
        
        engine = DynamicsEngine(
            world_width=200.0,
            world_height=200.0,
            obstacles=[],
            tick_rate=20.0,
            mobility_threshold=30.0,
            sensor_threshold=50.0,
        )
        
        # Execute EAST action
        for _ in range(20):
            state, collision = engine.step_unit(state, "EAST")
        
        # Speed should be accelerating toward 80% of max (8.0 m/s)
        target_speed = config.max_speed * 0.8
        assert state.target_speed == target_speed
        assert state.speed > 0.0
        assert state.speed <= target_speed
    
    def test_respects_max_turn_rate(self):
        """Test unit respects maximum turn rate."""
        state, config = self.create_test_unit()
        state.heading = 0.0  # Start facing EAST
        
        engine = DynamicsEngine(
            world_width=200.0,
            world_height=200.0,
            obstacles=[],
            tick_rate=20.0,
            mobility_threshold=30.0,
            sensor_threshold=50.0,
        )
        
        # Execute WEST action (target 180° - large turn)
        prev_heading = state.heading
        state, collision = engine.step_unit(state, "WEST")
        
        # Turn amount should not exceed max_turn_rate * dt
        max_turn_per_step = config.max_turn_rate * (1.0 / 20.0)
        heading_change = abs(state.heading - prev_heading)
        assert heading_change <= max_turn_per_step + 0.1  # Small tolerance
    
    def test_respects_max_acceleration(self):
        """Test unit respects maximum acceleration."""
        state, config = self.create_test_unit()
        state.speed = 0.0
        
        engine = DynamicsEngine(
            world_width=200.0,
            world_height=200.0,
            obstacles=[],
            tick_rate=20.0,
            mobility_threshold=30.0,
            sensor_threshold=50.0,
        )
        
        # Execute EAST action
        prev_speed = state.speed
        state, collision = engine.step_unit(state, "EAST")
        
        # Acceleration should not exceed max_accel * dt
        max_accel_per_step = config.max_accel * (1.0 / 20.0)
        speed_change = state.speed - prev_speed
        assert speed_change <= max_accel_per_step + 0.01  # Small tolerance


class TestEndToEndBehavior:
    """Test end-to-end behavior with high-level actions."""
    
    def test_unit_reaches_objective(self):
        """Test unit can navigate toward objective using high-level actions."""
        env = MissionGymEnv()
        obs, info = env.reset()
        
        initial_pos = (env.attackers[0].x, env.attackers[0].y)
        objective_pos = (env.objective.x, env.objective.y)
        initial_dist = math.hypot(
            objective_pos[0] - initial_pos[0],
            objective_pos[1] - initial_pos[1]
        )
        
        # Move toward objective for 50 steps
        for _ in range(50):
            # Simple heuristic: move in general direction
            dx = objective_pos[0] - env.attackers[0].x
            dy = objective_pos[1] - env.attackers[0].y
            
            # Choose action based on direction
            if abs(dx) > abs(dy):
                action_idx = 3 if dx > 0 else 7  # EAST or WEST
            else:
                action_idx = 1 if dy > 0 else 5  # NORTH or SOUTH
            
            action = np.array([action_idx, 0, 0, 0], dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action)
        
        final_pos = (env.attackers[0].x, env.attackers[0].y)
        final_dist = math.hypot(
            objective_pos[0] - final_pos[0],
            objective_pos[1] - final_pos[1]
        )
        
        # Distance should have decreased significantly
        assert final_dist < initial_dist - 20.0  # At least 20m closer
        
        env.close()
    
    def test_multi_unit_coordination(self):
        """Test multiple units can move independently."""
        env = MissionGymEnv()
        obs, info = env.reset()
        
        initial_positions = [(a.x, a.y) for a in env.attackers]
        
        # Each unit moves in different direction
        actions = [1, 3, 5, 7]  # N, E, S, W
        for _ in range(20):
            action = np.array(actions, dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action)
        
        final_positions = [(a.x, a.y) for a in env.attackers]
        
        # All units should have moved
        for i in range(4):
            dist_moved = math.hypot(
                final_positions[i][0] - initial_positions[i][0],
                final_positions[i][1] - initial_positions[i][1]
            )
            assert dist_moved > 5.0  # Moved at least 5 meters
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
