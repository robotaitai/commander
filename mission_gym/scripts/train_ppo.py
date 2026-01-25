#!/usr/bin/env python3
"""PPO training script for Mission Gym using Stable-Baselines3."""

# Suppress TensorFlow/TensorBoard noise (MUST be first, before ANY imports)
import os
# Set environment variables before any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no info, 2=no warnings, 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'

# Ensure CUDA is visible (clear any stale CUDA_VISIBLE_DEVICES)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import warnings
from io import StringIO

# Suppress all Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ImportWarning)
warnings.filterwarnings('ignore', message='.*tensorboard.*')
warnings.filterwarnings('ignore', message='.*matplotlib.*')

# Context manager to suppress stderr during noisy imports
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stderr = self._original_stderr
        return False

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
        help="Custom run name prefix (timestamp auto-appended, e.g. 'my-run' → 'my-run-20260121-143052')",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--subproc",
        action="store_true",
        help="Use SubprocVecEnv for true parallelism (recommended for n-envs > 4)",
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
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., 'runs/my-run/checkpoints/ppo_mission_100000_steps.zip')",
    )
    parser.add_argument(
        "--parent-checkpoint",
        type=str,
        default=None,
        help="Alias for --load-checkpoint. Loads parent checkpoint for policy branching.",
    )
    parser.add_argument(
        "--branch-name",
        type=str,
        default=None,
        help="Branch name for policy lineage (e.g., 'explore-v2'). Creates run name '<branch-name>-<timestamp>'",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes about this training run (saved in lineage.json)",
    )
    args = parser.parse_args()
    
    # Handle parent-checkpoint alias
    if args.parent_checkpoint and not args.load_checkpoint:
        args.load_checkpoint = args.parent_checkpoint
    
    # Import run utilities
    from mission_gym.scripts.run_utils import (
        create_run_dir, save_run_configs, save_run_metadata, save_rewards_history,
        print_banner, print_gpu_status, print_step, print_info, print_warning,
        print_error, print_success, print_divider, Colors, get_nvidia_smi_info,
        update_unified_dashboard, get_runs_dir,
        save_lineage, check_checkpoint_compatibility,
    )
    
    c = Colors
    
    # Determine run name (handle branching logic)
    run_name_input = None
    if args.branch_name and args.load_checkpoint:
        # Branch mode: use branch name
        run_name_input = args.branch_name
    elif args.run_name:
        # Custom run name
        run_name_input = args.run_name
    elif args.load_checkpoint and not args.branch_name:
        # Loading checkpoint without explicit name - add "branch" prefix
        run_name_input = "branch"
    # Otherwise None - will auto-generate
    
    # Create run directory
    run_dir = create_run_dir(run_name_input)
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
        # Suppress TensorBoard/TensorFlow errors during import
        with SuppressStderr():
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        print_info("Stable-Baselines3 imported")
    except ImportError as e:
        print_error(f"Import failed: {e}")
        print_info("Install with: pip install stable-baselines3")
        return 1
    
    try:
        from mission_gym.env import MissionGymEnv
        from mission_gym.scripts.monitoring import (
            HTMLMonitorCallback, EvalWithMonitorCallback, MetricsCallback,
            RichTrainingCallback,
        )
        print_info("MissionGymEnv and monitoring imported")
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return 1
    
    # Create vectorized environment
    print()
    print_step(3, f"Creating {args.n_envs} parallel environments")
    
    # Use config snapshot from run directory to ensure consistency
    cfg_dir = run_dir / "configs"
    
    def make_env(seed: int):
        def _init():
            env = MissionGymEnv(config_dir=cfg_dir)
            env.reset(seed=seed)
            return env
        return _init
    
    try:
        if args.subproc:
            envs = SubprocVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
            print_info(f"Created {args.n_envs} environments (SubprocVecEnv - true parallelism)")
        else:
            envs = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
            print_info(f"Created {args.n_envs} environments (DummyVecEnv)")
    except Exception as e:
        print_error(f"Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create evaluation environment
    print()
    print_step(4, "Creating evaluation environment")
    eval_env = MissionGymEnv(config_dir=cfg_dir)
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
    
    # Pass reward component configs to the monitor
    try:
        # Get component configs from the reward function
        component_configs = eval_env.reward_fn.get_component_configs()
        html_monitor.update_component_configs(component_configs)
        print_info(f"Tracking {len(component_configs)} reward components")
    except Exception as e:
        print_warning(f"Could not load reward components: {e}")
    
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
    
    # Metrics callback for TensorBoard KPIs and episode summaries
    # Print every 100 episodes to reduce verbosity
    metrics_callback = MetricsCallback(verbose=0, print_freq=100)
    
    # Rich training callback for beautiful console output
    # Print every 500 iterations to reduce verbosity (progress bar still updates continuously)
    rich_training = RichTrainingCallback(print_freq=500)
    
    callbacks = CallbackList([html_monitor, eval_callback, checkpoint_callback, metrics_callback, rich_training])
    print_info("HTML dashboard callback configured")
    print_info("Evaluation callback configured")
    print_info("Checkpoint callback configured")
    print_info("Metrics callback configured (TensorBoard KPIs)")
    print_info("Rich training logger configured")
    
    # Save lineage and check compatibility
    print()
    print_step(6, "Saving policy lineage and checking compatibility")
    
    # Get observation and action space signatures
    test_env = envs.envs[0] if hasattr(envs, 'envs') else envs
    obs_space = test_env.observation_space
    action_space = test_env.action_space
    
    # Check compatibility if loading checkpoint
    if args.load_checkpoint:
        checkpoint_path_obj = Path(args.load_checkpoint)
        
        # Auto-append .zip if not present and file doesn't exist
        if not checkpoint_path_obj.exists() and not str(checkpoint_path_obj).endswith('.zip'):
            checkpoint_with_zip = Path(str(checkpoint_path_obj) + '.zip')
            if checkpoint_with_zip.exists():
                checkpoint_path_obj = checkpoint_with_zip
                args.load_checkpoint = str(checkpoint_path_obj)  # Update the args for later use
        
        if not checkpoint_path_obj.exists():
            print_error(f"Checkpoint not found: {checkpoint_path_obj}")
            return 1
        
        print_info(f"Checking checkpoint compatibility...")
        is_compatible, error_msg = check_checkpoint_compatibility(
            str(checkpoint_path_obj),
            obs_space,
            action_space,
        )
        
        if not is_compatible:
            print_error(f"Checkpoint incompatibility detected!")
            print_error(error_msg)
            print_warning("To fix this, either:")
            print_warning("  1. Revert config changes to match parent checkpoint")
            print_warning("  2. Train a new policy from scratch (remove --load-checkpoint)")
            return 1
        
        print_info("✓ Checkpoint is compatible")
    
    # Save lineage information
    save_lineage(
        run_dir=run_dir,
        parent_checkpoint=args.load_checkpoint,
        branch_name=args.branch_name,
        notes=args.notes,
        obs_space=obs_space,
        action_space=action_space,
    )
    print_info(f"Lineage saved to {run_dir / 'lineage.json'}")
    
    # Detect GPU and set device
    print()
    print_step(7, "Detecting compute device")
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print_info(f"✓ GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
            print_info(f"  CUDA version: {torch.version.cuda}")
            print_info(f"  Using device: cuda:0")
        else:
            device = "cpu"
            print_warning("⚠️  No GPU detected, using CPU (training will be slower)")
            print_info("  Tip: Check CUDA installation with 'nvidia-smi'")
    except Exception as e:
        device = "cpu"
        print_warning(f"⚠️  GPU detection failed: {e}")
        print_info("  Falling back to CPU")
    
    # Create PPO model with MlpPolicy (vector-only observations)
    print()
    print_step(8, "Creating PPO model with MlpPolicy")
    try:
        tb_log = None if args.no_tensorboard else str(log_path)
        
        # Optimize batch size based on n_envs for better GPU utilization
        # n_steps * n_envs should be divisible by batch_size
        rollout_buffer_size = 2048 // args.n_envs * args.n_envs
        batch_size = min(256, rollout_buffer_size // 4)  # Larger batches for GPU
        
        model = PPO(
            "MlpPolicy",
            envs,
            verbose=0,  # Disabled - using RichTrainingCallback instead
            seed=args.seed,
            tensorboard_log=tb_log,
            device=device,  # Explicit device selection
            learning_rate=3e-4,
            n_steps=2048 // args.n_envs,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={
                "net_arch": [256, 256],  # Shared layers for pi and vf
            },
        )
        print_info(f"PPO model created (MlpPolicy, vector-only obs, device={device})")
        print_info(f"  Rollout buffer: {rollout_buffer_size}, Batch size: {batch_size}")
    except Exception as e:
        print_error(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print()
        print_step(9, f"Loading parent checkpoint")
        try:
            model = PPO.load(str(args.load_checkpoint), env=envs, device=device)
            print_info(f"✓ Parent checkpoint loaded: {Path(args.load_checkpoint).name}")
            print_info(f"  Model moved to device: {device}")
            # Extract timesteps from checkpoint name if possible
            checkpoint_name = Path(args.load_checkpoint).stem
            if "_steps" in checkpoint_name:
                try:
                    loaded_steps = int(checkpoint_name.split("_steps")[0].split("_")[-1])
                    print_info(f"  Parent trained for {loaded_steps:,} timesteps")
                except:
                    pass
        except Exception as e:
            print_error(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Training info
    print()
    print_divider()
    # Update unified dashboard to include this run
    update_unified_dashboard()
    unified_dashboard = get_runs_dir() / "dashboard.html"
    
    print()
    print(f"  {c.colorize('📊 Monitoring:', c.BOLD, c.BRIGHT_CYAN)}")
    print()
    print(f"     {c.colorize('All Runs:', c.BRIGHT_CYAN)} {unified_dashboard}")
    print(f"     {c.colorize('           ', c.DIM)} Master dashboard with run selector")
    print(f"     {c.colorize('This Run:', c.BRIGHT_BLUE)} {dashboard_path}")
    print(f"     {c.colorize('           ', c.DIM)} Auto-refreshes every 5s")
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
    print_step(10, "Saving results")
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
    
    # Update unified dashboard after training
    update_unified_dashboard()
    
    # Final summary
    print()
    print_divider()
    print()
    print(f"  {c.colorize('✅ Training Complete!', c.BOLD, c.BRIGHT_GREEN)}")
    print()
    print(f"  {c.colorize('📁 Run Directory:', c.BRIGHT_BLUE)} {run_dir}")
    print(f"  {c.colorize('📊 All Runs:', c.BRIGHT_CYAN)} {unified_dashboard}")
    print(f"  {c.colorize('📊 This Run:', c.BRIGHT_BLUE)} {dashboard_path}")
    print(f"  {c.colorize('🎮 To evaluate:', c.BRIGHT_BLUE)} python -m mission_gym.scripts.evaluate --model {model_path}")
    print()
    print_divider()
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
