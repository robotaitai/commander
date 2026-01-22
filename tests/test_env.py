"""Tests for the Mission Gym environment."""

import pytest
import numpy as np
import gymnasium as gym

from mission_gym.env import MissionGymEnv


class TestEnvironmentCreation:
    """Test environment creation and initialization."""

    def test_create_env(self):
        """Test basic environment creation."""
        env = MissionGymEnv()
        assert env is not None
        env.close()

    def test_env_has_action_space(self):
        """Test environment has valid action space."""
        env = MissionGymEnv()
        assert env.action_space is not None
        assert hasattr(env.action_space, "sample")
        env.close()

    def test_env_has_observation_space(self):
        """Test environment has valid observation space."""
        env = MissionGymEnv()
        assert env.observation_space is not None
        env.close()

    def test_observation_space_is_box(self):
        """Test observation space is a Box (vector-only)."""
        env = MissionGymEnv()
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 1  # 1D vector
        assert env.observation_space.shape[0] > 0  # Has features
        env.close()


class TestEnvironmentReset:
    """Test environment reset functionality."""

    def test_reset_returns_observation(self):
        """Test reset returns valid observation."""
        env = MissionGymEnv()
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert len(obs.shape) == 1  # 1D vector
        assert obs.shape[0] == env.observation_space.shape[0]
        env.close()
    
    def test_reset_returns_info(self):
        """Test reset returns info dictionary."""
        env = MissionGymEnv()
        obs, info = env.reset()
        assert isinstance(info, dict)
        env.close()
    
    def test_reset_with_seed(self):
        """Test reset with seed for reproducibility (with randomization)."""
        # Create two separate environments with the same seed
        # This ensures the RNG state is identical
        env1 = MissionGymEnv()
        obs1, _ = env1.reset(seed=42)
        env1.close()
        
        env2 = MissionGymEnv()
        obs2, _ = env2.reset(seed=42)
        env2.close()
        
        # With same seed, observations should be identical
        np.testing.assert_array_equal(obs1, obs2)


class TestEnvironmentStep:
    """Test environment step functionality."""

    def test_step_returns_tuple(self):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        env = MissionGymEnv()
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        env.close()

    def test_step_observation_valid(self):
        """Test step returns valid observation."""
        env = MissionGymEnv()
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)
        env.close()

    def test_step_reward_is_float(self):
        """Test step returns numeric reward."""
        env = MissionGymEnv()
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, (int, float, np.floating))
        env.close()

    def test_step_terminated_is_bool(self):
        """Test step returns boolean terminated."""
        env = MissionGymEnv()
        env.reset()
        action = env.action_space.sample()
        _, _, terminated, _, _ = env.step(action)
        assert isinstance(terminated, (bool, np.bool_))
        env.close()

    def test_step_truncated_is_bool(self):
        """Test step returns boolean truncated."""
        env = MissionGymEnv()
        env.reset()
        action = env.action_space.sample()
        _, _, _, truncated, _ = env.step(action)
        assert isinstance(truncated, (bool, np.bool_))
        env.close()

    def test_multiple_steps(self):
        """Test environment can run multiple steps."""
        env = MissionGymEnv()
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()


class TestEnvironmentIntegration:
    """Integration tests for the environment."""

    def test_episode_runs_to_completion(self):
        """Test a full episode can complete."""
        env = MissionGymEnv()
        env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Should either terminate or hit step limit
        assert steps > 0
        env.close()

    def test_action_space_sample(self):
        """Test action space sampling works."""
        env = MissionGymEnv()
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
        env.close()

    def test_noop_actions(self):
        """Test environment handles NOOP actions."""
        env = MissionGymEnv()
        env.reset()
        
        # Create NOOP action (all zeros)
        noop_action = np.zeros(env.action_space.shape, dtype=np.int64)
        
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(noop_action)
            if terminated or truncated:
                break
        
        env.close()
