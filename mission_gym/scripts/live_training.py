#!/usr/bin/env python3
"""
Training with periodic live visualization.

Opens a pygame window to show the simulation every N episodes,
so you can visually monitor agent progress during training.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def main():
    """Run training with live visualization."""
    parser = argparse.ArgumentParser(
        description="Train PPO with periodic live visualization"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--render-freq",
        type=int,
        default=50,
        help="Show live visualization every N episodes (default: 50)",
    )
    parser.add_argument(
        "--render-steps",
        type=int,
        default=200,
        help="Steps to show per visualization (default: 200)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="ppo_mission_gym",
        help="Path to save model (default: ppo_mission_gym)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mission Gym - Training with Live Visualization")
    print("=" * 60)
    
    try:
        import pygame
    except ImportError:
        print("Error: pygame is required. Install with: pip install pygame")
        return 1
    
    # Imports
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from mission_gym.env import MissionGymEnv
    
    # Create training environments
    def make_env(seed):
        def _init():
            env = MissionGymEnv()
            env.reset(seed=seed)
            return env
        return _init
    
    print(f"\n[1] Creating {args.n_envs} training environments...")
    envs = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    
    # Create visualization environment
    print("[2] Creating visualization environment...")
    vis_env = MissionGymEnv(render_mode="human")
    
    # Custom callback for periodic visualization
    class LiveVisCallback(BaseCallback):
        def __init__(self, vis_env, render_freq, render_steps, verbose=1):
            super().__init__(verbose)
            self.vis_env = vis_env
            self.render_freq = render_freq
            self.render_steps = render_steps
            self.episode_count = 0
            self.last_render_episode = -render_freq  # Render on first episode
        
        def _on_step(self) -> bool:
            # Count episodes
            for done in self.locals.get("dones", []):
                if done:
                    self.episode_count += 1
                    
                    # Check if time to render
                    if self.episode_count - self.last_render_episode >= self.render_freq:
                        self.last_render_episode = self.episode_count
                        self._run_visualization()
            
            return True
        
        def _run_visualization(self):
            """Run a visualization episode."""
            print(f"\n🎮 Live visualization (episode {self.episode_count})...")
            
            obs, _ = self.vis_env.reset()
            total_reward = 0.0
            
            for step in range(self.render_steps):
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("   Skipping visualization (ESC)")
                            return True
                
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.vis_env.step(action)
                total_reward += reward
                
                self.vis_env.render()
                time.sleep(0.03)  # Slow down for visibility
                
                if terminated or truncated:
                    break
            
            print(f"   Reward: {total_reward:.2f}, Steps: {step + 1}")
            return True
    
    # Create model
    print("\n[3] Creating PPO model...")
    model = PPO(
        "MultiInputPolicy",
        envs,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=2048 // args.n_envs,
        batch_size=64,
    )
    
    # Create callback
    vis_callback = LiveVisCallback(vis_env, args.render_freq, args.render_steps)
    
    # Train
    print(f"\n[4] Training for {args.timesteps} timesteps...")
    print(f"    Live visualization every {args.render_freq} episodes")
    print(f"    Press ESC during visualization to skip")
    print()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=vis_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    # Save model
    print(f"\n[5] Saving model to {args.save_path}...")
    model.save(args.save_path)
    
    # Cleanup
    envs.close()
    vis_env.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
