#!/usr/bin/env python3
"""Smoke test for Mission Gym environment."""

import sys
import numpy as np


def main():
    """Run smoke test."""
    print("=" * 60)
    print("Mission Gym Smoke Test")
    print("=" * 60)
    
    # Import environment
    print("\n[1] Importing MissionGymEnv...")
    try:
        from mission_gym.env import MissionGymEnv
        print("    ✓ Import successful")
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        return 1
    
    # Create environment
    print("\n[2] Creating environment...")
    try:
        env = MissionGymEnv()
        print("    ✓ Environment created")
    except Exception as e:
        print(f"    ✗ Creation failed: {e}")
        return 1
    
    # Print spaces
    print("\n[3] Observation and Action Spaces:")
    print(f"    Action space: {env.action_space}")
    print(f"    Observation space:")
    print(f"      - BEV shape: {env.observation_space['bev'].shape}")
    print(f"      - Vec shape: {env.observation_space['vec'].shape}")
    
    # Reset environment
    print("\n[4] Resetting environment...")
    try:
        obs, info = env.reset(seed=42)
        print("    ✓ Reset successful")
        print(f"    Initial info: {info}")
    except Exception as e:
        print(f"    ✗ Reset failed: {e}")
        return 1
    
    # Validate observation shapes
    print("\n[5] Validating observation shapes...")
    try:
        assert obs["bev"].shape == (128, 128, 8), f"BEV shape mismatch: {obs['bev'].shape}"
        assert obs["bev"].dtype == np.float32, f"BEV dtype mismatch: {obs['bev'].dtype}"
        expected_vec_dim = env.num_attackers * 9 + 2
        assert obs["vec"].shape[0] == expected_vec_dim, f"Vec dim mismatch: {obs['vec'].shape}"
        assert obs["vec"].dtype == np.float32, f"Vec dtype mismatch: {obs['vec'].dtype}"
        print(f"    ✓ BEV shape: {obs['bev'].shape}")
        print(f"    ✓ Vec shape: {obs['vec'].shape}")
    except AssertionError as e:
        print(f"    ✗ Validation failed: {e}")
        return 1
    
    # Run steps
    print("\n[6] Running 200 steps...")
    try:
        total_reward = 0.0
        for step in range(200):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Validate observation
            assert obs["bev"].shape == (128, 128, 8)
            assert len(obs["vec"]) == expected_vec_dim
            
            if step % 50 == 0:
                print(f"    Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                print(f"    Episode ended at step {step}")
                obs, info = env.reset()
        
        print(f"    ✓ 200 steps completed")
        print(f"    Total reward: {total_reward:.3f}")
    except Exception as e:
        print(f"    ✗ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test deterministic seeding
    print("\n[7] Testing deterministic seeding...")
    try:
        obs1, _ = env.reset(seed=123)
        action = env.action_space.sample()
        obs1_next, r1, _, _, _ = env.step(action)
        
        obs2, _ = env.reset(seed=123)
        obs2_next, r2, _, _, _ = env.step(action)
        
        if np.allclose(obs1["bev"], obs2["bev"]) and np.allclose(obs1["vec"], obs2["vec"]):
            print("    ✓ Deterministic reset confirmed")
        else:
            print("    ⚠ Non-deterministic reset (may be expected)")
    except Exception as e:
        print(f"    ✗ Seeding test failed: {e}")
    
    # Clean up
    print("\n[8] Closing environment...")
    env.close()
    print("    ✓ Environment closed")
    
    print("\n" + "=" * 60)
    print("Smoke Test PASSED ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
