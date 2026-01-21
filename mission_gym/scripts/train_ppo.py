#!/usr/bin/env python3
"""PPO training script for Mission Gym using Stable-Baselines3."""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    """Run PPO training."""
    parser = argparse.ArgumentParser(description="Train PPO agent on Mission Gym")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="ppo_mission_gym",
        help="Path to save the trained model (default: ppo_mission_gym)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for tensorboard logs (default: ./logs)",
    )
    parser.add_argument(
        "--html-dashboard",
        type=str,
        default="training_dashboard.html",
        help="Path for HTML dashboard (default: training_dashboard.html)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency in timesteps (default: 5000)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable tensorboard logging",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mission Gym - PPO Training")
    print("=" * 60)
    
    # Initialize pygame for snapshot capture (headless)
    try:
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Headless mode
        import pygame
        pygame.init()
        pygame.font.init()
    except Exception:
        pass  # Snapshots will be disabled if pygame fails
    
    # Import dependencies
    print("\n[1] Importing dependencies...")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
        from stable_baselines3.common.vec_env import DummyVecEnv
        print("    ✓ Stable-Baselines3 imported")
    except ImportError as e:
        print(f"    ✗ Import failed: {e}")
        print("    Install with: pip install stable-baselines3")
        return 1
    
    try:
        from mission_gym.env import MissionGymEnv
        from mission_gym.scripts.monitoring import HTMLMonitorCallback, EvalWithMonitorCallback
        print("    ✓ MissionGymEnv and monitoring imported")
    except ImportError as e:
        print(f"    ✗ Import failed: {e}")
        return 1
    
    # Create vectorized environment
    print(f"\n[2] Creating {args.n_envs} parallel environments...")
    
    def make_env(seed: int):
        def _init():
            env = MissionGymEnv()
            env.reset(seed=seed)
            return env
        return _init
    
    try:
        envs = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
        print(f"    ✓ Created {args.n_envs} environments")
    except Exception as e:
        print(f"    ✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create evaluation environment
    print("\n[3] Creating evaluation environment...")
    eval_env = MissionGymEnv()
    eval_env.reset(seed=args.seed + 1000)
    print("    ✓ Evaluation environment created")
    
    # Create callbacks
    print("\n[4] Setting up callbacks...")
    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # HTML monitoring callback
    html_monitor = HTMLMonitorCallback(
        html_path=args.html_dashboard,
        update_freq=500,
        verbose=1,
    )
    
    # Evaluation callback with HTML integration
    eval_callback = EvalWithMonitorCallback(
        eval_env=eval_env,
        html_monitor=html_monitor,
        n_eval_episodes=5,
        eval_freq=args.eval_freq,
        verbose=1,
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // args.n_envs, 1),
        save_path=str(log_path / "checkpoints"),
        name_prefix="ppo_mission",
    )
    
    callbacks = CallbackList([html_monitor, eval_callback, checkpoint_callback])
    print("    ✓ HTML dashboard callback configured")
    print("    ✓ Evaluation callback configured")
    print("    ✓ Checkpoint callback configured")
    
    # Create PPO model with MultiInputPolicy (CNN on BEV + MLP on vec)
    print("\n[5] Creating PPO model with MultiInputPolicy...")
    try:
        tb_log = None if args.no_tensorboard else str(log_path)
        
        model = PPO(
            "MultiInputPolicy",
            envs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tb_log,
            learning_rate=3e-4,
            n_steps=2048 // args.n_envs,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={
                "net_arch": {
                    "pi": [256, 256],
                    "vf": [256, 256],
                },
            },
        )
        print("    ✓ PPO model created")
    except Exception as e:
        print(f"    ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Training info
    print(f"\n[6] Starting training for {args.timesteps} timesteps...")
    print()
    print("    📊 Monitoring options:")
    print(f"       • HTML Dashboard: {args.html_dashboard}")
    print(f"         Open in browser and it auto-refreshes every 30s")
    if not args.no_tensorboard:
        print(f"       • TensorBoard: tensorboard --logdir {args.log_dir}")
        print(f"         Then open http://localhost:6006")
    print()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        print("\n    ✓ Training completed!")
    except KeyboardInterrupt:
        print("\n    Training interrupted by user")
    except Exception as e:
        print(f"\n    ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save final model
    print(f"\n[7] Saving model to {args.save_path}...")
    try:
        model.save(args.save_path)
        print(f"    ✓ Model saved to {args.save_path}.zip")
    except Exception as e:
        print(f"    ✗ Save failed: {e}")
    
    # Cleanup
    envs.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\n📊 View final dashboard: {args.html_dashboard}")
    print(f"\n🎮 To test the trained agent:")
    print(f"   python -m mission_gym.scripts.evaluate --model {args.save_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
