"""Tests for the episode metrics tracking module."""

import pytest
import math
from dataclasses import dataclass


class TestEpisodeMetrics:
    """Test EpisodeMetrics dataclass."""

    def test_to_dict_basic(self):
        """Test basic conversion to dictionary."""
        from mission_gym.metrics import EpisodeMetrics
        
        metrics = EpisodeMetrics(
            win=True,
            terminated_reason="captured",
            episode_steps=100,
            episode_sim_time=10.0,
        )
        
        d = metrics.to_dict()
        
        assert d["win"] is True
        assert d["terminated_reason"] == "captured"
        assert d["episode_steps"] == 100
        assert d["episode_sim_time"] == 10.0

    def test_to_dict_derived_rates(self):
        """Test derived rates are calculated correctly."""
        from mission_gym.metrics import EpisodeMetrics
        
        metrics = EpisodeMetrics(
            tag_attempts_attacker=10,
            tag_hits_attacker=7,
            tag_attempts_defender=5,
            tag_hits_defender=2,
            distance_by_unit=[100.0, 150.0, 200.0],
            episode_sim_time=60.0,
            detected_time=30.0,
        )
        
        d = metrics.to_dict()
        
        assert d["tag_hit_rate_attacker"] == pytest.approx(0.7)
        assert d["tag_hit_rate_defender"] == pytest.approx(0.4)
        assert d["distance_total"] == pytest.approx(450.0)
        assert d["distance_mean"] == pytest.approx(150.0)
        assert d["detected_time_pct"] == pytest.approx(50.0)

    def test_to_dict_zero_division(self):
        """Test derived rates handle zero division."""
        from mission_gym.metrics import EpisodeMetrics
        
        metrics = EpisodeMetrics(
            tag_attempts_attacker=0,
            tag_attempts_defender=0,
            episode_sim_time=0.0,
        )
        
        d = metrics.to_dict()
        
        assert d["tag_hit_rate_attacker"] == 0.0
        assert d["tag_hit_rate_defender"] == 0.0
        assert d["detected_time_pct"] == 0.0


class TestMetricsTracker:
    """Test MetricsTracker class."""

    def test_reset(self):
        """Test reset initializes metrics correctly."""
        from mission_gym.metrics import MetricsTracker
        
        tracker = MetricsTracker(num_attackers=3)
        tracker.reset()
        
        assert len(tracker.ep.distance_by_unit) == 3
        assert all(d == 0.0 for d in tracker.ep.distance_by_unit)
        assert tracker.ep.episode_steps == 0

    def test_record_step_increments_counters(self):
        """Test record_step increments episode counters."""
        from mission_gym.metrics import MetricsTracker
        
        # Create mock attackers
        @dataclass
        class MockUnit:
            x: float
            y: float
            integrity: float
            is_disabled: bool = False
        
        @dataclass
        class MockObjective:
            capture_progress: float
            x: float = 50.0
            y: float = 50.0
            radius: float = 10.0
        
        tracker = MetricsTracker(num_attackers=2)
        tracker.reset()
        
        attackers = [MockUnit(10.0, 10.0, 100.0), MockUnit(20.0, 20.0, 100.0)]
        objective = MockObjective(capture_progress=0.5)
        
        tracker.record_step(
            attackers=attackers,
            defenders=[],
            objective=objective,
            dt=0.1,
            collisions=1,
            integrity_lost=10.0,
            units_disabled=0,
            any_detected=True,
            in_objective_zone=True,
        )
        
        assert tracker.ep.episode_steps == 1
        assert tracker.ep.episode_sim_time == pytest.approx(0.1)
        assert tracker.ep.collisions_total == 1
        assert tracker.ep.integrity_lost_total == pytest.approx(10.0)
        assert tracker.ep.detected_time == pytest.approx(0.1)
        assert tracker.ep.time_in_objective_zone == pytest.approx(0.1)
        assert tracker.ep.detection_events == 1

    def test_record_step_tracks_distance(self):
        """Test record_step tracks distance travelled."""
        from mission_gym.metrics import MetricsTracker
        
        @dataclass
        class MockUnit:
            x: float
            y: float
            integrity: float
            is_disabled: bool = False
        
        @dataclass
        class MockObjective:
            capture_progress: float
        
        tracker = MetricsTracker(num_attackers=1)
        tracker.reset()
        
        objective = MockObjective(capture_progress=0.0)
        
        # First step - establishes initial position
        tracker.record_step(
            attackers=[MockUnit(0.0, 0.0, 100.0)],
            defenders=[],
            objective=objective,
            dt=0.1,
            collisions=0,
            integrity_lost=0,
            units_disabled=0,
            any_detected=False,
            in_objective_zone=False,
        )
        
        # Second step - unit moved 3-4-5 triangle (distance = 5)
        tracker.record_step(
            attackers=[MockUnit(3.0, 4.0, 100.0)],
            defenders=[],
            objective=objective,
            dt=0.1,
            collisions=0,
            integrity_lost=0,
            units_disabled=0,
            any_detected=False,
            in_objective_zone=False,
        )
        
        assert tracker.ep.distance_by_unit[0] == pytest.approx(5.0)

    def test_detection_events_rising_edge(self):
        """Test detection events only count rising edges."""
        from mission_gym.metrics import MetricsTracker
        
        @dataclass
        class MockUnit:
            x: float
            y: float
            integrity: float
            is_disabled: bool = False
        
        @dataclass
        class MockObjective:
            capture_progress: float
        
        tracker = MetricsTracker(num_attackers=1)
        tracker.reset()
        
        attackers = [MockUnit(0.0, 0.0, 100.0)]
        objective = MockObjective(0.0)
        
        # Not detected -> detected (should count)
        tracker.record_step(attackers, [], objective, 0.1, 0, 0, 0, any_detected=True, in_objective_zone=False)
        assert tracker.ep.detection_events == 1
        
        # Detected -> detected (should NOT count)
        tracker.record_step(attackers, [], objective, 0.1, 0, 0, 0, any_detected=True, in_objective_zone=False)
        assert tracker.ep.detection_events == 1
        
        # Detected -> not detected (should NOT count)
        tracker.record_step(attackers, [], objective, 0.1, 0, 0, 0, any_detected=False, in_objective_zone=False)
        assert tracker.ep.detection_events == 1
        
        # Not detected -> detected (should count again)
        tracker.record_step(attackers, [], objective, 0.1, 0, 0, 0, any_detected=True, in_objective_zone=False)
        assert tracker.ep.detection_events == 2

    def test_finish(self):
        """Test finish returns correct dictionary."""
        from mission_gym.metrics import MetricsTracker
        
        @dataclass
        class MockUnit:
            x: float
            y: float
            integrity: float
            is_disabled: bool = False
        
        tracker = MetricsTracker(num_attackers=2)
        tracker.reset()
        tracker.ep.episode_sim_time = 30.0
        
        attackers = [
            MockUnit(0.0, 0.0, 100.0, is_disabled=False),
            MockUnit(10.0, 10.0, 0.0, is_disabled=True),
        ]
        
        result = tracker.finish(win=True, reason="captured", attackers=attackers)
        
        assert result["win"] is True
        assert result["terminated_reason"] == "captured"
        assert result["time_to_capture"] == pytest.approx(30.0)
        assert result["num_attackers_alive_end"] == 1
        assert result["num_attackers_disabled_total"] == 1


class TestEngagementStats:
    """Test engagement stats tracking."""

    def test_stats_reset(self):
        """Test EngagementStats reset."""
        from mission_gym.engagement import EngagementStats
        
        stats = EngagementStats()
        stats.tag_attempts_attacker = 5
        stats.tag_hits_attacker = 3
        
        stats.reset()
        
        assert stats.tag_attempts_attacker == 0
        assert stats.tag_hits_attacker == 0


class TestEnvMetricsIntegration:
    """Integration tests for metrics in the environment."""

    def test_episode_metrics_in_info(self):
        """Test that episode_metrics appears in info when episode ends."""
        from mission_gym.env import MissionGymEnv
        
        env = MissionGymEnv()
        env.reset(seed=42)
        
        # Run until episode ends
        done = False
        info = {}
        max_steps = 2000
        step = 0
        
        while not done and step < max_steps:
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        
        # Check episode_metrics is in final info
        assert "episode_metrics" in info, "episode_metrics should be in info when episode ends"
        
        metrics = info["episode_metrics"]
        assert "win" in metrics
        assert "terminated_reason" in metrics
        assert "episode_steps" in metrics
        assert "distance_total" in metrics
        assert "tag_hit_rate_attacker" in metrics
        
        env.close()

    def test_observation_space_vec_dimension(self):
        """Test observation space reflects updated vector dimension."""
        from mission_gym.env import MissionGymEnv
        
        env = MissionGymEnv()
        
        # Vector should have 10 features per attacker (including cos/sin heading) + 2 global
        expected_dim = env.num_attackers * 10 + 2
        actual_dim = env.observation_space.shape[0]
        
        assert actual_dim == expected_dim, f"Expected {expected_dim}, got {actual_dim}"
        
        env.close()
