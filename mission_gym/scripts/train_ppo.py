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
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (default: auto-generated like 'swift-falcon-20260121-143052')",
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
    
    # Import run utilities
    from mission_gym.scripts.run_utils import (
        create_run_dir, save_run_configs, save_run_metadata, save_rewards_history,
        print_banner, print_gpu_status, print_step, print_info, print_warning,
        print_error, print_success, print_divider, Colors, get_nvidia_smi_info,
    )
    
    c = Colors
    
    # Create run directory
    run_dir = create_run_dir(args.run_name)
    run_name = run_dir.name
    
    # Print banner
    print_banner(run_name)
    
    # Print GPU status
    print_gpu_status()
    print_divider()
    
    # Save configs
    print()
    print_step(1, "Saving configuration files")
    save_run_configs(run_dir)
    print_info(f"Configs saved to {run_dir / 'configs'}")
    
    # Save run metadata
    save_run_metadata(run_dir, vars(args))
    print_info(f"Metadata saved to {run_dir / 'run_metadata.json'}")
    
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
    print()
    print_step(2, "Importing dependencies")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
        from stable_baselines3.common.vec_env import DummyVecEnv
        print_info("Stable-Baselines3 imported")
    except ImportError as e:
        print_error(f"Import failed: {e}")
        print_info("Install with: pip install stable-baselines3")
        return 1
    
    try:
        from mission_gym.env import MissionGymEnv
        from mission_gym.scripts.monitoring import HTMLMonitorCallback, EvalWithMonitorCallback
        print_info("MissionGymEnv and monitoring imported")
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return 1
    
    # Create vectorized environment
    print()
    print_step(3, f"Creating {args.n_envs} parallel environments")
    
    def make_env(seed: int):
        def _init():
            env = MissionGymEnv()
            env.reset(seed=seed)
            return env
        return _init
    
    try:
        envs = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
        print_info(f"Created {args.n_envs} environments")
    except Exception as e:
        print_error(f"Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create evaluation environment
    print()
    print_step(4, "Creating evaluation environment")
    eval_env = MissionGymEnv()
    eval_env.reset(seed=args.seed + 1000)
    print_info("Evaluation environment created")
    
    # Create callbacks
    print()
    print_step(5, "Setting up callbacks")
    
    log_path = run_dir / "logs"
    checkpoint_path = run_dir / "checkpoints"
    dashboard_path = run_dir / "dashboard.html"
    
    # HTML monitoring callback with GPU stats
    html_monitor = HTMLMonitorCallback(
        html_path=str(dashboard_path),
        update_freq=500,
        verbose=0,
        run_dir=run_dir,
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
        save_path=str(checkpoint_path),
        name_prefix="ppo_mission",
    )
    
    callbacks = CallbackList([html_monitor, eval_callback, checkpoint_callback])
    print_info("HTML dashboard callback configured")
    print_info("Evaluation callback configured")
    print_info("Checkpoint callback configured")
    
    # Create PPO model with MultiInputPolicy (CNN on BEV + MLP on vec)
    print()
    print_step(6, "Creating PPO model with MultiInputPolicy")
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
        print_info("PPO model created")
    except Exception as e:
        print_error(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Training info
    print()
    print_divider()
    print()
    print(f"  {c.colorize('📊 Monitoring:', c.BOLD, c.BRIGHT_CYAN)}")
    print()
    print(f"     {c.colorize('Dashboard:', c.BRIGHT_BLUE)} {dashboard_path}")
    print(f"     {c.colorize('           ', c.DIM)} Open in browser - auto-refreshes every 30s")
    if not args.no_tensorboard:
        print(f"     {c.colorize('TensorBoard:', c.BRIGHT_BLUE)} tensorboard --logdir {log_path}")
        print(f"     {c.colorize('            ', c.DIM)} Then open http://localhost:6006")
    print()
    print_divider()
    print()
    print(f"  {c.colorize(f'🚀 Starting training for {args.timesteps:,} timesteps...', c.BOLD, c.BRIGHT_GREEN)}")
    print()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        print()
        print_success("Training completed!")
    except KeyboardInterrupt:
        print()
        print_warning("Training interrupted by user")
    except Exception as e:
        print()
        print_error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save rewards history
    print()
    print_step(7, "Saving results")
    save_rewards_history(
        run_dir,
        html_monitor.episode_rewards,
        html_monitor.timesteps_history,
        html_monitor.episode_lengths,
    )
    print_info(f"Rewards history saved to {run_dir / 'rewards_history.json'}")
    print_info(f"Summary saved to {run_dir / 'summary.txt'}")
    
    # Save final model
    model_path = run_dir / "final_model"
    try:
        model.save(str(model_path))
        print_info(f"Model saved to {model_path}.zip")
    except Exception as e:
        print_error(f"Save failed: {e}")
    
    # Cleanup
    envs.close()
    eval_env.close()
    
    # Final summary
    print()
    print_divider()
    print()
    print(f"  {c.colorize('✅ Training Complete!', c.BOLD, c.BRIGHT_GREEN)}")
    print()
    print(f"  {c.colorize('📁 Run Directory:', c.BRIGHT_BLUE)} {run_dir}")
    print(f"  {c.colorize('📊 Dashboard:', c.BRIGHT_BLUE)} {dashboard_path}")
    print(f"  {c.colorize('🎮 To evaluate:', c.BRIGHT_BLUE)} python -m mission_gym.scripts.evaluate --model {model_path}")
    print()
    print_divider()
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
