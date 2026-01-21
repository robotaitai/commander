"""Pygame renderer for visualization."""

from typing import Optional
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from mission_gym.config import FullConfig, Obstacle, ObjectiveConfig
from mission_gym.dynamics import UnitState
from mission_gym.scenario import ObjectiveState


# Colors
COLOR_BG = (30, 30, 40)
COLOR_GRID = (50, 50, 60)
COLOR_OBSTACLE = (80, 80, 100)
COLOR_OBJECTIVE = (60, 120, 60)
COLOR_OBJECTIVE_CAPTURED = (100, 200, 100)
COLOR_ATTACKER_UGV = (100, 150, 255)
COLOR_ATTACKER_UAV = (150, 200, 255)
COLOR_DEFENDER = (255, 100, 100)
COLOR_DISABLED = (100, 100, 100)
COLOR_TAG_BEAM = (255, 255, 100)
COLOR_TEXT = (220, 220, 220)
COLOR_SELECTED = (255, 255, 0)
# Range circle colors (semi-transparent effect via thin lines)
COLOR_RANGE_MAX = (255, 100, 100, 80)      # Red for max range
COLOR_RANGE_OPTIMAL = (100, 255, 100, 80)  # Green for optimal range
COLOR_RANGE_ATTACKER_MAX = (100, 150, 255)
COLOR_RANGE_ATTACKER_OPT = (150, 200, 100)
COLOR_RANGE_DEFENDER_MAX = (255, 100, 100)
COLOR_RANGE_DEFENDER_OPT = (255, 180, 100)


class Renderer:
    """Pygame-based renderer for the environment."""
    
    def __init__(
        self,
        config: FullConfig,
        window_size: tuple[int, int] = (800, 800),
    ):
        self.config = config
        self.window_size = window_size
        self.world_width = config.world.width
        self.world_height = config.world.height
        
        # Scale factors
        self.scale_x = window_size[0] / self.world_width
        self.scale_y = window_size[1] / self.world_height
        self.scale = min(self.scale_x, self.scale_y)
        
        # Offset to center the world
        self.offset_x = (window_size[0] - self.world_width * self.scale) / 2
        self.offset_y = (window_size[1] - self.world_height * self.scale) / 2
        
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.selected_unit: int = 0
        
    def initialize(self) -> None:
        """Initialize pygame and create window."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is not installed")
        
        pygame.init()
        pygame.display.set_caption("Mission Gym")
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
    
    def close(self) -> None:
        """Close the renderer."""
        if PYGAME_AVAILABLE and pygame.get_init():
            pygame.quit()
    
    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(self.offset_x + x * self.scale)
        screen_y = int(self.offset_y + (self.world_height - y) * self.scale)  # Flip Y
        return screen_x, screen_y
    
    def render(
        self,
        attackers: list[UnitState],
        defenders: list[UnitState],
        objective: ObjectiveState,
        time_remaining: float,
        selected_unit: int = 0,
    ) -> None:
        """Render the current state."""
        if self.screen is None:
            return
        
        self.selected_unit = selected_unit
        
        # Clear screen
        self.screen.fill(COLOR_BG)
        
        # Draw grid
        self._draw_grid()
        
        # Draw obstacles
        for obs in self.config.world.obstacles:
            self._draw_obstacle(obs)
        
        # Draw objective
        self._draw_objective(objective)
        
        # Draw units
        for defender in defenders:
            self._draw_unit(defender, is_attacker=False)
        
        for i, attacker in enumerate(attackers):
            self._draw_unit(attacker, is_attacker=True, is_selected=(i == selected_unit))
        
        # Draw HUD
        self._draw_hud(attackers, defenders, objective, time_remaining)
        
        pygame.display.flip()
    
    def _draw_grid(self) -> None:
        """Draw background grid."""
        grid_spacing = 20  # meters
        
        for x in range(0, int(self.world_width) + 1, grid_spacing):
            start = self.world_to_screen(x, 0)
            end = self.world_to_screen(x, self.world_height)
            pygame.draw.line(self.screen, COLOR_GRID, start, end, 1)
        
        for y in range(0, int(self.world_height) + 1, grid_spacing):
            start = self.world_to_screen(0, y)
            end = self.world_to_screen(self.world_width, y)
            pygame.draw.line(self.screen, COLOR_GRID, start, end, 1)
    
    def _draw_obstacle(self, obs: Obstacle) -> None:
        """Draw an obstacle."""
        if obs.type == "circle":
            center = self.world_to_screen(obs.x, obs.y)
            radius = int(obs.radius * self.scale)
            pygame.draw.circle(self.screen, COLOR_OBSTACLE, center, radius)
        elif obs.type == "rect":
            corners = obs.get_corners()
            if corners:
                screen_corners = [self.world_to_screen(x, y) for x, y in corners]
                pygame.draw.polygon(self.screen, COLOR_OBSTACLE, screen_corners)
    
    def _draw_objective(self, objective: ObjectiveState) -> None:
        """Draw the objective zone."""
        center = self.world_to_screen(objective.x, objective.y)
        radius = int(objective.radius * self.scale)
        
        # Fill based on capture progress
        progress = objective.capture_progress / objective.capture_time_required
        color = (
            int(COLOR_OBJECTIVE[0] + (COLOR_OBJECTIVE_CAPTURED[0] - COLOR_OBJECTIVE[0]) * progress),
            int(COLOR_OBJECTIVE[1] + (COLOR_OBJECTIVE_CAPTURED[1] - COLOR_OBJECTIVE[1]) * progress),
            int(COLOR_OBJECTIVE[2] + (COLOR_OBJECTIVE_CAPTURED[2] - COLOR_OBJECTIVE[2]) * progress),
        )
        
        pygame.draw.circle(self.screen, color, center, radius)
        pygame.draw.circle(self.screen, COLOR_TEXT, center, radius, 2)
        
        # Draw progress text
        progress_text = f"{progress * 100:.0f}%"
        text_surface = self.small_font.render(progress_text, True, COLOR_TEXT)
        text_rect = text_surface.get_rect(center=center)
        self.screen.blit(text_surface, text_rect)
    
    def _draw_range_circles(
        self,
        center: tuple[int, int],
        is_attacker: bool,
        is_selected: bool,
    ) -> None:
        """Draw range circles showing max and optimal tag range."""
        eng = self.config.engagement
        
        # Calculate screen radii
        max_range_px = int(eng.tag_range * self.scale)
        optimal_range_px = int(eng.tag_optimal_range * self.scale)
        
        # Choose colors based on team
        if is_attacker:
            max_color = COLOR_RANGE_ATTACKER_MAX
            opt_color = COLOR_RANGE_ATTACKER_OPT
        else:
            max_color = COLOR_RANGE_DEFENDER_MAX
            opt_color = COLOR_RANGE_DEFENDER_OPT
        
        # Adjust alpha/thickness based on selection
        if is_selected:
            max_thickness = 2
            opt_thickness = 2
        else:
            max_thickness = 1
            opt_thickness = 1
            # Dim the colors for non-selected units
            max_color = tuple(c // 3 for c in max_color)
            opt_color = tuple(c // 3 for c in opt_color)
        
        # Draw max range circle (dashed effect via multiple arcs)
        if max_range_px > 5:
            # Draw dashed circle for max range
            for angle in range(0, 360, 20):
                start_angle = math.radians(angle)
                end_angle = math.radians(angle + 10)
                # Draw arc segment
                rect = (
                    center[0] - max_range_px,
                    center[1] - max_range_px,
                    max_range_px * 2,
                    max_range_px * 2,
                )
                pygame.draw.arc(self.screen, max_color, rect, start_angle, end_angle, max_thickness)
        
        # Draw optimal range circle (solid)
        if optimal_range_px > 5:
            pygame.draw.circle(self.screen, opt_color, center, optimal_range_px, opt_thickness)
    
    def _draw_unit(
        self,
        unit: UnitState,
        is_attacker: bool,
        is_selected: bool = False,
    ) -> None:
        """Draw a unit."""
        center = self.world_to_screen(unit.x, unit.y)
        
        # Determine color
        if unit.is_disabled:
            color = COLOR_DISABLED
        elif is_attacker:
            color = COLOR_ATTACKER_UAV if unit.category == "air" else COLOR_ATTACKER_UGV
        else:
            color = COLOR_DEFENDER
        
        # Draw range circles (only for active units)
        if not unit.is_disabled:
            self._draw_range_circles(center, is_attacker, is_selected)
        
        # Draw unit body
        radius = int((unit.type_config.radius if unit.type_config else 1.0) * self.scale * 1.5)
        radius = max(radius, 6)
        
        # Draw selection ring
        if is_selected:
            pygame.draw.circle(self.screen, COLOR_SELECTED, center, radius + 4, 2)
        
        # Draw altitude indicator for air units
        if unit.category == "air" and unit.altitude > 0:
            # Draw shadow
            shadow_offset = unit.altitude * 3
            shadow_center = (center[0] + shadow_offset, center[1] + shadow_offset)
            pygame.draw.circle(self.screen, (20, 20, 30), shadow_center, radius)
        
        pygame.draw.circle(self.screen, color, center, radius)
        
        # Draw heading indicator
        heading_rad = math.radians(unit.heading)
        dx = math.cos(heading_rad) * radius * 1.5
        dy = -math.sin(heading_rad) * radius * 1.5  # Flip Y for screen coords
        end_point = (int(center[0] + dx), int(center[1] + dy))
        pygame.draw.line(self.screen, COLOR_TEXT, center, end_point, 2)
        
        # Draw integrity bar
        if not unit.is_disabled:
            bar_width = radius * 2
            bar_height = 4
            bar_x = center[0] - bar_width // 2
            bar_y = center[1] - radius - 8
            
            # Background
            pygame.draw.rect(
                self.screen, (60, 60, 60),
                (bar_x, bar_y, bar_width, bar_height)
            )
            
            # Integrity
            max_integrity = unit.type_config.initial_integrity if unit.type_config else 100.0
            integrity_pct = max(0.0, min(1.0, unit.integrity / max_integrity))
            integrity_width = max(0, int(bar_width * integrity_pct))
            integrity_color = (
                min(255, max(0, int(255 * (1 - integrity_pct)))),
                min(255, max(0, int(255 * integrity_pct))),
                0,
            )
            if integrity_width > 0:
                pygame.draw.rect(
                    self.screen, integrity_color,
                    (bar_x, bar_y, integrity_width, bar_height)
                )
        
        # Draw unit ID
        id_text = str(unit.unit_id)
        text_surface = self.small_font.render(id_text, True, COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(center[0], center[1] + radius + 10))
        self.screen.blit(text_surface, text_rect)
    
    def _draw_hud(
        self,
        attackers: list[UnitState],
        defenders: list[UnitState],
        objective: ObjectiveState,
        time_remaining: float,
    ) -> None:
        """Draw heads-up display."""
        # Time remaining
        minutes = int(time_remaining) // 60
        seconds = int(time_remaining) % 60
        time_text = f"Time: {minutes:02d}:{seconds:02d}"
        text_surface = self.font.render(time_text, True, COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        # Capture progress
        progress = objective.capture_progress / objective.capture_time_required
        progress_text = f"Capture: {progress * 100:.1f}%"
        text_surface = self.font.render(progress_text, True, COLOR_TEXT)
        self.screen.blit(text_surface, (10, 35))
        
        # Selected unit info
        if 0 <= self.selected_unit < len(attackers):
            unit = attackers[self.selected_unit]
            info_lines = [
                f"Selected: {unit.unit_type} (ID: {unit.unit_id})",
                f"Position: ({unit.x:.1f}, {unit.y:.1f})",
                f"Speed: {unit.speed:.1f} m/s",
                f"Heading: {unit.heading:.0f}°",
                f"Integrity: {unit.integrity:.0f}%",
            ]
            if unit.category == "air":
                info_lines.append(f"Altitude: {unit.altitude}")
            if unit.tag_cooldown > 0:
                info_lines.append(f"Tag CD: {unit.tag_cooldown:.1f}s")
            
            y_offset = self.window_size[1] - len(info_lines) * 22 - 10
            for line in info_lines:
                text_surface = self.small_font.render(line, True, COLOR_TEXT)
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 22
        
        # Controls hint
        controls = "1-4: Select | WASD: Move | T: Tag | ESC: Quit"
        text_surface = self.small_font.render(controls, True, (150, 150, 150))
        self.screen.blit(text_surface, (self.window_size[0] - 320, 10))
    
    def tick(self, fps: int = 60) -> None:
        """Limit framerate."""
        if self.clock:
            self.clock.tick(fps)
    
    def get_events(self) -> list:
        """Get pygame events."""
        if PYGAME_AVAILABLE:
            return pygame.event.get()
        return []
