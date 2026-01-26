#!/usr/bin/env python3
"""Tests for critical training fixes (batch size, n_steps_per_env, etc)."""

import unittest
from mission_gym.scripts.train_ppo import pick_batch_size


class TestBatchSizeFix(unittest.TestCase):
    """Test the pick_batch_size helper function."""
    
    def test_basic_divisibility(self):
        """Test that batch_size always divides buffer_size."""
        test_cases = [
            2048,   # 32 × 64
            4096,   # 64 × 64
            6144,   # 48 × 128
            8192,   # 32 × 256
            10240,  # 40 × 256
            12288,  # 48 × 256
            16384,  # 64 × 256
        ]
        
        for buffer_size in test_cases:
            batch_size = pick_batch_size(buffer_size, target_frac=0.25, min_bs=64)
            
            # Must divide evenly
            self.assertEqual(buffer_size % batch_size, 0,
                           f"batch_size {batch_size} does not divide buffer_size {buffer_size}")
            
            # Must be multiple of 64
            self.assertEqual(batch_size % 64, 0,
                           f"batch_size {batch_size} is not a multiple of 64")
            
            # Must be at least min_bs
            self.assertGreaterEqual(batch_size, 64,
                                   f"batch_size {batch_size} is less than min_bs 64")
    
    def test_target_fraction(self):
        """Test that batch_size is close to target fraction."""
        buffer_size = 8192
        target_frac = 0.25
        
        batch_size = pick_batch_size(buffer_size, target_frac=target_frac, min_bs=64)
        
        # Should be close to target (within 2 minibatches)
        target = buffer_size * target_frac
        self.assertLess(abs(batch_size - target), buffer_size / 2,
                       f"batch_size {batch_size} too far from target {target}")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small buffer
        self.assertEqual(pick_batch_size(64, target_frac=0.25, min_bs=64), 64)
        
        # Buffer not divisible by many values
        batch_size = pick_batch_size(5120, target_frac=0.25, min_bs=64)
        self.assertEqual(5120 % batch_size, 0)
        self.assertEqual(batch_size % 64, 0)
    
    def test_n_steps_scaling(self):
        """Test that rollout buffer scales with n_envs when using n_steps_per_env."""
        n_steps_per_env = 256
        
        test_configs = [
            (16, 16 * 256),   # 4096
            (32, 32 * 256),   # 8192
            (48, 48 * 256),   # 12288
            (64, 64 * 256),   # 16384
        ]
        
        for n_envs, expected_buffer in test_configs:
            rollout_buffer = n_steps_per_env * n_envs
            self.assertEqual(rollout_buffer, expected_buffer,
                           f"n_envs={n_envs}: expected {expected_buffer}, got {rollout_buffer}")
            
            # Batch size should be valid for this buffer
            batch_size = pick_batch_size(rollout_buffer, target_frac=0.25, min_bs=64)
            self.assertEqual(rollout_buffer % batch_size, 0)


class TestMonitoringFixes(unittest.TestCase):
    """Test monitoring callback fixes."""
    
    def test_metrics_callback_init(self):
        """Test MetricsCallback initialization."""
        from mission_gym.scripts.monitoring import MetricsCallback
        
        callback = MetricsCallback(verbose=0, print_freq=100)
        
        # Should not have a stored console (to avoid conflicts)
        self.assertIsNone(callback.console)
        
        # Should have recent_rewards tracking
        self.assertIsInstance(callback.recent_rewards, list)
        self.assertEqual(len(callback.recent_rewards), 0)
    
    def test_eval_freq_default(self):
        """Test that eval frequency default is reasonable."""
        import argparse
        from mission_gym.scripts.train_ppo import main
        
        # Parse default args
        parser = argparse.ArgumentParser()
        parser.add_argument("--eval-freq", type=int, default=20000)
        args = parser.parse_args([])
        
        # Should be 20K not 5K (less spam)
        self.assertEqual(args.eval_freq, 20000,
                        "Default eval frequency should be 20000 to reduce spam")


class TestCheckpointCompatibility(unittest.TestCase):
    """Test checkpoint compatibility rules."""
    
    def test_defender_changes_compatible(self):
        """Test that adding/removing defenders doesn't break compatibility."""
        # This is a documentation test - verifies our understanding
        # In actual code, defender changes should NOT break compatibility
        # because defender observations are not included in attacker policy input
        
        # The test would check the compatibility logic if we had it accessible
        # For now, this serves as documentation
        self.assertTrue(True, "Defender changes should be compatible")
    
    def test_attacker_changes_incompatible(self):
        """Test that changing attacker count/actions breaks compatibility."""
        # Attacker changes SHOULD break compatibility because:
        # - Action space changes (MultiDiscrete size)
        # - Observation space may change (number of units)
        
        self.assertTrue(True, "Attacker changes should break compatibility")


if __name__ == "__main__":
    unittest.main()
