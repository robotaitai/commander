#!/usr/bin/env python3
"""
View the current scenario configuration.

Shows:
- World map with obstacles
- Unit spawn positions
- Objective zone
- Unit type stats
"""

import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def print_config_summary():
    """Print a text summary of the configuration."""
    from mission_gym.config import FullConfig
    
    config = FullConfig.load()
    
    print("=" * 70)
    print("MISSION GYM - SCENARIO OVERVIEW")
    print("=" * 70)
    
    # World
    print("\n🌍 WORLD")
    print("-" * 40)
    print(f"  Arena Size: {config.world.width} x {config.world.height} meters")
    print(f"  Physics Rate: {config.world.tick_rate} Hz")
    print(f"  Action Repeat: {config.world.action_repeat} (commands at {config.world.tick_rate/config.world.action_repeat} Hz)")
    print(f"  Max Episode: {config.world.max_duration} seconds ({config.world.max_duration/60:.1f} minutes)")
    print(f"  Obstacles: {len(config.world.obstacles)}")
    for i, obs in enumerate(config.world.obstacles):
        if obs.type == "circle":
            print(f"    [{i}] Circle at ({obs.x}, {obs.y}), radius={obs.radius}m")
        else:
            print(f"    [{i}] Rect at ({obs.x}, {obs.y}), {obs.width}x{obs.height}m, angle={obs.angle}°")
    
    # Objective
    print("\n🎯 OBJECTIVE")
    print("-" * 40)
    obj = config.scenario.objective
    print(f"  Position: ({obj.x}, {obj.y})")
    print(f"  Radius: {obj.radius}m")
    print(f"  Capture Time Required: {obj.capture_time_required} seconds")
    
    # Attackers
    print("\n🔵 ATTACKERS")
    print("-" * 40)
    for i, spawn in enumerate(config.scenario.attackers):
        unit_type = config.attacker_types.get(spawn.unit_type)
        if unit_type:
            category = "🚗" if unit_type.category == "ground" else "🚁"
            print(f"  [{i}] {category} {spawn.unit_type}")
            print(f"      Spawn: ({spawn.x}, {spawn.y}), heading={spawn.heading}°")
            if unit_type.category == "air":
                print(f"      Altitude: {spawn.altitude}")
            print(f"      Speed: {unit_type.max_speed} m/s, Accel: {unit_type.max_accel} m/s²")
            print(f"      Turn Rate: {unit_type.max_turn_rate}°/s")
            print(f"      Integrity: {unit_type.initial_integrity}")
            print(f"      Sensors: {', '.join(unit_type.sensors)}")
            print(f"      Actions: {len(unit_type.actions)} ({', '.join(unit_type.actions[:5])}...)")
    
    # Defenders
    print("\n🔴 DEFENDERS")
    print("-" * 40)
    for i, spawn in enumerate(config.scenario.defenders):
        unit_type = config.defender_types.get(spawn.unit_type)
        if unit_type:
            category = "🚗" if unit_type.category == "ground" else "🚁"
            print(f"  [{i}] {category} {spawn.unit_type}")
            print(f"      Spawn: ({spawn.x}, {spawn.y}), heading={spawn.heading}°")
            print(f"      Speed: {unit_type.max_speed} m/s")
            print(f"      Integrity: {unit_type.initial_integrity}")
            if spawn.patrol_waypoints:
                print(f"      Patrol: {len(spawn.patrol_waypoints)} waypoints")
                for wp in spawn.patrol_waypoints[:3]:
                    print(f"        → ({wp[0]}, {wp[1]})")
    
    # Engagement
    print("\n⚔️ ENGAGEMENT")
    print("-" * 40)
    eng = config.engagement
    print(f"  Max Tag Range: {eng.tag_range}m")
    print(f"  Optimal Range: {eng.tag_optimal_range}m (full damage)")
    print(f"  Tag FOV: ±{eng.tag_fov}°")
    print(f"  Damage at optimal: {eng.tag_damage}")
    print(f"  Damage at max range: {eng.tag_min_damage} (falloff)")
    print(f"  Tag Cooldown: {eng.tag_cooldown}s")
    print(f"  Mobility Degradation at: {eng.mobility_threshold}% integrity")
    print(f"  Sensor Degradation at: {eng.sensor_threshold}% integrity")
    print(f"  Disabled at: {eng.disabled_threshold}% integrity")
    
    # Reward
    print("\n💰 REWARD")
    print("-" * 40)
    rw = config.reward
    print(f"  Capture Progress: +{rw.capture_progress}/sec")
    print(f"  Win Bonus: +{rw.win_bonus}")
    print(f"  Time Penalty: {rw.time_penalty}/step")
    print(f"  Collision Penalty: {rw.collision_penalty}")
    print(f"  Integrity Loss: {rw.integrity_loss_penalty}/point")
    print(f"  Unit Disabled: {rw.unit_disabled_penalty}")
    if rw.enable_detected_penalty:
        print(f"  Detected Penalty: {rw.detected_time_penalty}/step")
    
    print("\n" + "=" * 70)


def show_map():
    """Show a visual map using pygame."""
    if not PYGAME_AVAILABLE:
        print("\nPygame not available for visual map. Install with: pip install pygame")
        return
    
    import math
    from mission_gym.config import FullConfig
    
    config = FullConfig.load()
    
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    
    size = 700
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("Mission Gym - Scenario Map")
    
    world_w = config.world.width
    world_h = config.world.height
    scale = (size - 40) / max(world_w, world_h)
    offset = 20
    
    def w2s(x, y):
        return int(offset + x * scale), int(offset + (world_h - y) * scale)
    
    # Colors
    BG = (20, 25, 30)
    GRID = (40, 45, 50)
    OBSTACLE = (80, 85, 100)
    OBJECTIVE = (40, 100, 40)
    ATTACKER = (80, 130, 220)
    DEFENDER = (220, 80, 80)
    TEXT = (180, 180, 180)
    WAYPOINT = (150, 80, 80)
    
    font = pygame.font.Font(None, 20)
    title_font = pygame.font.Font(None, 28)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
        
        screen.fill(BG)
        
        # Grid
        for x in range(0, int(world_w) + 1, 20):
            pygame.draw.line(screen, GRID, w2s(x, 0), w2s(x, world_h), 1)
        for y in range(0, int(world_h) + 1, 20):
            pygame.draw.line(screen, GRID, w2s(0, y), w2s(world_w, y), 1)
        
        # Border
        pygame.draw.rect(screen, TEXT, (offset, offset, int(world_w * scale), int(world_h * scale)), 2)
        
        # Obstacles
        for obs in config.world.obstacles:
            if obs.type == "circle":
                pygame.draw.circle(screen, OBSTACLE, w2s(obs.x, obs.y), int(obs.radius * scale))
            elif obs.type == "rect":
                corners = obs.get_corners()
                if corners:
                    pygame.draw.polygon(screen, OBSTACLE, [w2s(x, y) for x, y in corners])
        
        # Objective zone
        obj = config.scenario.objective
        pygame.draw.circle(screen, OBJECTIVE, w2s(obj.x, obj.y), int(obj.radius * scale))
        pygame.draw.circle(screen, (80, 180, 80), w2s(obj.x, obj.y), int(obj.radius * scale), 2)
        txt = font.render("OBJECTIVE", True, (120, 200, 120))
        screen.blit(txt, (w2s(obj.x, obj.y)[0] - txt.get_width()//2, w2s(obj.x, obj.y)[1] - 8))
        
        # Defender patrol waypoints
        for spawn in config.scenario.defenders:
            if spawn.patrol_waypoints:
                pts = [w2s(wp[0], wp[1]) for wp in spawn.patrol_waypoints]
                if len(pts) > 1:
                    pygame.draw.lines(screen, WAYPOINT, True, pts, 1)
                for pt in pts:
                    pygame.draw.circle(screen, WAYPOINT, pt, 3)
        
        # Get engagement ranges
        max_range_px = int(config.engagement.tag_range * scale)
        opt_range_px = int(config.engagement.tag_optimal_range * scale)
        
        # Attackers
        for i, spawn in enumerate(config.scenario.attackers):
            unit_type = config.attacker_types.get(spawn.unit_type)
            pos = w2s(spawn.x, spawn.y)
            radius = int((unit_type.radius if unit_type else 1.0) * scale * 2)
            radius = max(radius, 8)
            
            # Draw range circles
            pygame.draw.circle(screen, (60, 80, 120), pos, max_range_px, 1)  # Max range (dashed effect)
            pygame.draw.circle(screen, (80, 120, 80), pos, opt_range_px, 1)  # Optimal range
            
            # Draw unit
            pygame.draw.circle(screen, ATTACKER, pos, radius)
            
            # Heading arrow
            heading_rad = math.radians(spawn.heading)
            dx = math.cos(heading_rad) * radius * 1.5
            dy = -math.sin(heading_rad) * radius * 1.5
            pygame.draw.line(screen, (200, 220, 255), pos, (int(pos[0] + dx), int(pos[1] + dy)), 2)
            
            # Label
            label = f"A{i}: {spawn.unit_type}"
            txt = font.render(label, True, (150, 180, 255))
            screen.blit(txt, (pos[0] - txt.get_width()//2, pos[1] + radius + 5))
        
        # Defenders
        for i, spawn in enumerate(config.scenario.defenders):
            unit_type = config.defender_types.get(spawn.unit_type)
            pos = w2s(spawn.x, spawn.y)
            radius = int((unit_type.radius if unit_type else 1.0) * scale * 2)
            radius = max(radius, 8)
            
            # Draw range circles
            pygame.draw.circle(screen, (120, 60, 60), pos, max_range_px, 1)  # Max range
            pygame.draw.circle(screen, (140, 100, 60), pos, opt_range_px, 1)  # Optimal range
            
            # Draw unit
            pygame.draw.circle(screen, DEFENDER, pos, radius)
            
            # Heading arrow
            heading_rad = math.radians(spawn.heading)
            dx = math.cos(heading_rad) * radius * 1.5
            dy = -math.sin(heading_rad) * radius * 1.5
            pygame.draw.line(screen, (255, 150, 150), pos, (int(pos[0] + dx), int(pos[1] + dy)), 2)
            
            # Label
            label = f"D{i}: {spawn.unit_type}"
            txt = font.render(label, True, (255, 150, 150))
            screen.blit(txt, (pos[0] - txt.get_width()//2, pos[1] + radius + 5))
        
        # Title
        title = title_font.render(f"Scenario: {config.scenario.name}", True, TEXT)
        screen.blit(title, (offset, 5))
        
        # Legend - Row 1
        legend_y = size - 40
        pygame.draw.circle(screen, ATTACKER, (offset + 10, legend_y), 6)
        screen.blit(font.render("Attacker", True, ATTACKER), (offset + 22, legend_y - 7))
        pygame.draw.circle(screen, DEFENDER, (offset + 100, legend_y), 6)
        screen.blit(font.render("Defender", True, DEFENDER), (offset + 112, legend_y - 7))
        pygame.draw.circle(screen, OBJECTIVE, (offset + 200, legend_y), 6)
        screen.blit(font.render("Objective", True, (80, 180, 80)), (offset + 212, legend_y - 7))
        pygame.draw.rect(screen, OBSTACLE, (offset + 300, legend_y - 6, 12, 12))
        screen.blit(font.render("Obstacle", True, OBSTACLE), (offset + 318, legend_y - 7))
        
        # Legend - Row 2 (Range circles)
        legend_y2 = size - 20
        pygame.draw.circle(screen, (80, 120, 80), (offset + 10, legend_y2), 8, 1)
        screen.blit(font.render(f"Optimal ({config.engagement.tag_optimal_range}m)", True, (80, 120, 80)), (offset + 22, legend_y2 - 7))
        pygame.draw.circle(screen, (60, 80, 120), (offset + 180, legend_y2), 10, 1)
        screen.blit(font.render(f"Max Range ({config.engagement.tag_range}m)", True, (80, 100, 140)), (offset + 195, legend_y2 - 7))
        
        # Instructions
        instr = font.render("Press ESC or Q to close", True, (100, 100, 100))
        screen.blit(instr, (size - instr.get_width() - 10, 5))
        
        pygame.display.flip()
        pygame.time.wait(50)
    
    pygame.quit()


def main():
    """View scenario configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View Mission Gym scenario configuration")
    parser.add_argument("--no-map", action="store_true", help="Skip visual map, text only")
    args = parser.parse_args()
    
    # Print text summary
    print_config_summary()
    
    # Show visual map
    if not args.no_map:
        print("\nOpening visual map... (press ESC or Q to close)")
        show_map()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
