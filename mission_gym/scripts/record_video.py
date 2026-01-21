#!/usr/bin/env python3
"""Record simulation episodes as video/GIF files."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def main():
    """Record episodes to video/GIF."""
    parser = argparse.ArgumentParser(description="Record Mission Gym episodes as video")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (if None, uses random actions)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="episode",
        help="Output filename prefix (default: episode)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["gif", "mp4", "frames"],
        default="gif",
        help="Output format: gif, mp4, or frames (default: gif)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=600,
        help="Video size in pixels (default: 600)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mission Gym - Episode Recorder")
    print("=" * 60)
    
    if not PYGAME_AVAILABLE:
        print("Error: pygame is required. Install with: pip install pygame")
        return 1
    
    # Check for video libraries
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False
        print("Warning: imageio not installed. Install with: pip install imageio[ffmpeg]")
        if args.format != "frames":
            print("         Falling back to saving individual frames.")
            args.format = "frames"
    
    # Import environment
    from mission_gym.env import MissionGymEnv
    from mission_gym.renderer import Renderer
    
    # Load model if provided
    model = None
    if args.model:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(args.model)
            print(f"✓ Loaded model: {args.model}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("  Using random actions instead")
    
    # Create environment (no render mode - we'll render manually)
    env = MissionGymEnv()
    
    # Create custom renderer with specific size
    renderer = Renderer(env.config, window_size=(args.size, args.size))
    
    # Initialize pygame for offscreen rendering
    pygame.init()
    # Create an offscreen surface
    screen = pygame.Surface((args.size, args.size))
    
    output_dir = Path("recordings")
    output_dir.mkdir(exist_ok=True)
    
    for ep in range(args.episodes):
        print(f"\nRecording episode {ep + 1}/{args.episodes}...")
        
        obs, info = env.reset(seed=args.seed + ep)
        frames = []
        done = False
        step = 0
        total_reward = 0.0
        
        while not done and step < args.max_steps:
            # Get action
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
            
            # Render to surface
            frame = render_frame(
                screen, env, renderer, args.size,
                step, total_reward, info
            )
            frames.append(frame)
            
            if step % 100 == 0:
                print(f"  Step {step}, reward: {total_reward:.2f}")
        
        print(f"  Episode finished: {step} steps, reward: {total_reward:.2f}")
        
        # Save video/frames
        output_path = save_recording(
            frames, output_dir, args.output, ep, args.format, args.fps, HAS_IMAGEIO
        )
        print(f"  ✓ Saved: {output_path}")
    
    env.close()
    pygame.quit()
    
    print("\n" + "=" * 60)
    print(f"Recording complete! Files saved to: {output_dir}/")
    print("=" * 60)
    
    return 0


def render_frame(
    screen: "pygame.Surface",
    env,
    renderer,
    size: int,
    step: int,
    total_reward: float,
    info: dict,
) -> np.ndarray:
    """Render a single frame to numpy array."""
    import pygame
    
    # Colors
    BG = (30, 30, 40)
    GRID = (50, 50, 60)
    OBSTACLE = (80, 80, 100)
    OBJECTIVE = (60, 120, 60)
    OBJECTIVE_CAPTURED = (100, 200, 100)
    ATTACKER_UGV = (100, 150, 255)
    ATTACKER_UAV = (150, 200, 255)
    DEFENDER = (255, 100, 100)
    DISABLED = (100, 100, 100)
    TEXT = (220, 220, 220)
    
    config = env.config
    world_w = config.world.width
    world_h = config.world.height
    scale = size / max(world_w, world_h)
    
    def world_to_screen(x, y):
        return int(x * scale), int((world_h - y) * scale)
    
    # Clear
    screen.fill(BG)
    
    # Grid
    for x in range(0, int(world_w) + 1, 20):
        pygame.draw.line(screen, GRID, world_to_screen(x, 0), world_to_screen(x, world_h), 1)
    for y in range(0, int(world_h) + 1, 20):
        pygame.draw.line(screen, GRID, world_to_screen(0, y), world_to_screen(world_w, y), 1)
    
    # Obstacles
    for obs in config.world.obstacles:
        if obs.type == "circle":
            cx, cy = world_to_screen(obs.x, obs.y)
            pygame.draw.circle(screen, OBSTACLE, (cx, cy), int(obs.radius * scale))
        elif obs.type == "rect":
            corners = obs.get_corners()
            if corners:
                pts = [world_to_screen(x, y) for x, y in corners]
                pygame.draw.polygon(screen, OBSTACLE, pts)
    
    # Objective
    obj = env.objective
    ox, oy = world_to_screen(obj.x, obj.y)
    progress = obj.capture_progress / obj.capture_time_required
    color = tuple(int(OBJECTIVE[i] + (OBJECTIVE_CAPTURED[i] - OBJECTIVE[i]) * progress) for i in range(3))
    pygame.draw.circle(screen, color, (ox, oy), int(obj.radius * scale))
    pygame.draw.circle(screen, TEXT, (ox, oy), int(obj.radius * scale), 2)
    
    # Units
    import math
    for unit in env.defenders + env.attackers:
        ux, uy = world_to_screen(unit.x, unit.y)
        
        if unit.is_disabled:
            color = DISABLED
        elif unit.team == "attacker":
            color = ATTACKER_UAV if unit.category == "air" else ATTACKER_UGV
        else:
            color = DEFENDER
        
        radius = int((unit.type_config.radius if unit.type_config else 1.0) * scale * 1.5)
        radius = max(radius, 5)
        
        # Shadow for air units
        if unit.category == "air" and unit.altitude > 0:
            offset = unit.altitude * 3
            pygame.draw.circle(screen, (20, 20, 30), (ux + offset, uy + offset), radius)
        
        pygame.draw.circle(screen, color, (ux, uy), radius)
        
        # Heading
        heading_rad = math.radians(unit.heading)
        dx = math.cos(heading_rad) * radius * 1.5
        dy = -math.sin(heading_rad) * radius * 1.5
        pygame.draw.line(screen, TEXT, (ux, uy), (int(ux + dx), int(uy + dy)), 2)
        
        # Integrity bar
        if not unit.is_disabled:
            bar_w = radius * 2
            bar_x = ux - bar_w // 2
            bar_y = uy - radius - 6
            integrity = unit.integrity / 100.0
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, 3))
            pygame.draw.rect(screen, (int(255 * (1 - integrity)), int(255 * integrity), 0),
                           (bar_x, bar_y, int(bar_w * integrity), 3))
    
    # HUD
    font = pygame.font.Font(None, 24)
    
    # Step counter
    text = font.render(f"Step: {step}", True, TEXT)
    screen.blit(text, (10, 10))
    
    # Reward
    text = font.render(f"Reward: {total_reward:.1f}", True, TEXT)
    screen.blit(text, (10, 35))
    
    # Capture progress
    text = font.render(f"Capture: {progress * 100:.0f}%", True, TEXT)
    screen.blit(text, (10, 60))
    
    # Convert to numpy array
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))  # pygame uses (width, height), we want (height, width)
    
    return frame


def save_recording(
    frames: list,
    output_dir: Path,
    prefix: str,
    episode: int,
    format: str,
    fps: int,
    has_imageio: bool,
) -> Path:
    """Save recorded frames."""
    if format == "frames":
        # Save individual frames
        frame_dir = output_dir / f"{prefix}_ep{episode}_frames"
        frame_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            import imageio
            imageio.imwrite(frame_dir / f"frame_{i:04d}.png", frame)
        return frame_dir
    
    elif format == "gif":
        import imageio
        output_path = output_dir / f"{prefix}_ep{episode}.gif"
        # Subsample for smaller GIF
        step = max(1, len(frames) // 150)
        subsampled = frames[::step]
        imageio.mimsave(output_path, subsampled, fps=fps // step, loop=0)
        return output_path
    
    elif format == "mp4":
        import imageio
        output_path = output_dir / f"{prefix}_ep{episode}.mp4"
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return output_path
    
    return output_dir


if __name__ == "__main__":
    sys.exit(main())
