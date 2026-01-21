#!/usr/bin/env python3
"""Manual play script for Mission Gym with keyboard controls."""

import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame is required for manual play. Install with: pip install pygame")
    sys.exit(1)

from mission_gym.env import MissionGymEnv


def main():
    """Run manual play mode."""
    print("=" * 60)
    print("Mission Gym - Manual Play")
    print("=" * 60)
    print()
    print("Controls:")
    print("  1-4     : Select attacker unit")
    print("  W/S     : Throttle up/down")
    print("  A/D     : Turn left/right")
    print("  SPACE   : Brake")
    print("  H       : Hold position")
    print("  T       : Tag (attempt to disable nearby defender)")
    print("  E       : Scan")
    print("  R/F     : Altitude up/down (UAV only)")
    print("  ESC     : Quit")
    print()
    
    # Create environment with rendering
    env = MissionGymEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    
    selected_unit = 0
    env.set_selected_unit(selected_unit)
    
    # Action mapping for keyboard
    # Default to NOOP for all units
    running = True
    clock = pygame.time.Clock()
    
    print("Starting game...")
    print(f"Number of attackers: {env.num_attackers}")
    print(f"Number of defenders: {env.num_defenders}")
    
    episode_reward = 0.0
    episode_steps = 0
    
    while running:
        # Handle events
        action_override = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Unit selection
                elif event.key == pygame.K_1:
                    selected_unit = 0
                    env.set_selected_unit(selected_unit)
                elif event.key == pygame.K_2:
                    selected_unit = min(1, env.num_attackers - 1)
                    env.set_selected_unit(selected_unit)
                elif event.key == pygame.K_3:
                    selected_unit = min(2, env.num_attackers - 1)
                    env.set_selected_unit(selected_unit)
                elif event.key == pygame.K_4:
                    selected_unit = min(3, env.num_attackers - 1)
                    env.set_selected_unit(selected_unit)
        
        # Get keyboard state for continuous input
        keys = pygame.key.get_pressed()
        
        # Determine action for selected unit
        unit_action = 0  # NOOP
        
        if keys[pygame.K_w]:
            unit_action = 1  # THROTTLE_UP
        elif keys[pygame.K_s]:
            unit_action = 2  # THROTTLE_DOWN
        elif keys[pygame.K_a]:
            unit_action = 3  # TURN_LEFT / YAW_LEFT
        elif keys[pygame.K_d]:
            unit_action = 4  # TURN_RIGHT / YAW_RIGHT
        elif keys[pygame.K_SPACE]:
            unit_action = 5  # BRAKE
        elif keys[pygame.K_h]:
            unit_action = 6  # HOLD
        elif keys[pygame.K_t]:
            unit_action = 7  # TAG
        elif keys[pygame.K_e]:
            unit_action = 8  # SCAN
        elif keys[pygame.K_r]:
            # ALT_UP for UAV (need to check unit type)
            if selected_unit < env.num_attackers:
                attacker = env.attackers[selected_unit]
                if attacker.category == "air":
                    unit_action = 5  # ALT_UP is at index 5 for UAV
        elif keys[pygame.K_f]:
            # ALT_DOWN for UAV
            if selected_unit < env.num_attackers:
                attacker = env.attackers[selected_unit]
                if attacker.category == "air":
                    unit_action = 6  # ALT_DOWN is at index 6 for UAV
        
        # Build action array
        action = [0] * env.num_attackers  # NOOP for all
        action[selected_unit] = unit_action
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1
        
        # Render
        env.render()
        
        # Check episode end
        if terminated:
            print()
            print("=" * 40)
            print("VICTORY! Objective captured!")
            print(f"Episode reward: {episode_reward:.2f}")
            print(f"Episode steps: {episode_steps}")
            print("=" * 40)
            
            # Wait a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
        
        elif truncated:
            print()
            print("=" * 40)
            print("Episode ended (time limit or all disabled)")
            print(f"Episode reward: {episode_reward:.2f}")
            print(f"Episode steps: {episode_steps}")
            print("=" * 40)
            
            pygame.time.wait(2000)
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
        
        # Limit framerate
        clock.tick(60)
    
    env.close()
    print("\nGame ended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
