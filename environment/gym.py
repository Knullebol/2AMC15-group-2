from gymnasium import spaces
from TUeMap import draw_TUe_map
from TUeMap import Buildings

import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects

class TUeMapEnv(gym.Env):
    """
    A gymnasium continuous environment integrating TUe map.
    """
    def __init__(self):
        super(TUeMapEnv, self).__init__()
        self.width, self.height = 1280, 860
        
        # Action: 0=forward, 1=turn left, 2=turn right
        self.action_space = spaces.Discrete(3)
        
        # State: x, y, theta (all continuous)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi], dtype=np.float32),
            high=np.array([self.width, self.height, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        self.screen = None
        self.map_surface = None
        self.access_mask = None  # 1=road, 0=obstacle
        self._preprocess_map()
        self.path = []
        
        # Movement parameters
        self.forward_speed = 5.0  # pixels per step
        self.turn_speed = np.deg2rad(10)  # radians per step
        
        self.delivery_points = self.add_delivery_points()

    def _preprocess_map(self):
        """
        Mask buildings and roads to mark accessible areas and obstacles.
        """
        pygame.init()
        self.map_surface = draw_TUe_map(pygame, self.map_surface)
        
        map_pixels = pygame.surfarray.array3d(self.map_surface)  # shape: (width, height, 3)
        map_pixels = map_pixels.transpose(2, 0, 1)  # shape: (3, width, height)
        
        # Accessible: road color (180,180,180) or street color (120,120,120)
        road_colors = [(180,180,180), (120,120,120)]
        mask = np.zeros((self.width, self.height), dtype=bool)
        
        for color in road_colors:
            color_arr = np.array(color)[:, None, None]
            matches = np.all(map_pixels == color_arr, axis=0)
            mask = np.logical_or(mask, matches)
            
        self.access_mask = mask.astype(np.uint8)

    def reset(self, seed=None):
        """
        Reset the environment to a random accessible location (not obstacle).
        """
        super().reset(seed=seed)

        while True:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            px = int(x * (self.width-1))
            py = int(y * (self.height-1))
            
            if self.access_mask[px, py]:
                theta = np.random.uniform(-np.pi, np.pi)
                self.state = np.array([px, py, theta], dtype=np.float32)
                break
            
        self.path = [self.state[:2].copy()]
        return self.state.copy(), {}

    def step(self, action):
        """
        Take a step in the environment based on the action.
        """
        x, y, theta = self.state
        
        # Apply action
        if action == 0:
            # Move forward
            nx = x + self.forward_speed * np.cos(theta)
            ny = y + self.forward_speed * np.sin(theta)
            ntheta = theta
            
        elif action == 1:
            # Turn left
            nx, ny = x, y
            ntheta = theta + self.turn_speed
            
        elif action == 2:
            # Turn right
            nx, ny = x, y
            ntheta = theta - self.turn_speed
            
        # Normalize theta to [-pi, pi]
        ntheta = (ntheta + np.pi) % (2 * np.pi) - np.pi
        
        # Prevent going out of bounds
        nx = np.clip(nx, 0, self.width-1)
        ny = np.clip(ny, 0, self.height-1)
        
        # Only update position if accessible
        px, py = int(nx), int(ny)
        if self.access_mask[px, py]:
            self.state = np.array([nx, ny, ntheta], dtype=np.float32)
            reward = 0.0
        else:
            # If not accessible, stay in place, only update theta if turning
            if action == 0:
                # Forward blocked
                self.state = np.array([x, y, theta], dtype=np.float32)
            else:
                self.state = np.array([x, y, ntheta], dtype=np.float32)
            reward = -5.0  # Penalty for hitting obstacle
        terminated = False
        truncated = False
        self.path.append(self.state[:2].copy())
        return self.state.copy(), reward, terminated, truncated, {}

    def add_delivery_points(self, points=[], n_points=2):
        """
        Add multiple delivery points to the environment.
        """
        if not points:
            n = 0
            while n < n_points:
                x = int(np.random.uniform(0, 1) * (self.width-1))
                y = int(np.random.uniform(0, 1) * (self.height-1))
                if self.access_mask[x, y]:
                    points.append((x, y))
                    n += 1
        return points
    
    def render(self):
        """
        Display the TUe map (Obsolete).
        """
        if self.map_surface is None:
            self._preprocess_map()
            
        if self.screen is None:
            pygame.init()
            self.screen = draw_TUe_map(pygame, self.screen)

        # Draw agent
        x = int(self.state[0] * (self.width-1))
        y = int(self.state[1] * (self.height-1))
        font = pygame.font.SysFont('calibri', 20, bold=True)
        running = True
        clicked_coord = None
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    clicked_coord = (mouse_x, mouse_y)
            # Redraw map and agent every frame
            self.screen.blit(self.map_surface, (0,0))
            pygame.draw.circle(self.screen, (233, 113, 50), (x, y), 10)
            # Draw coordinates at top right if clicked
            if clicked_coord is not None:
                coord_text = f"({clicked_coord[0]}, {clicked_coord[1]})"
                text_surface = font.render(coord_text, True, (0,0,0))
                text_rect = text_surface.get_rect(topright=(self.width - 10, 10))
                pygame.draw.rect(self.screen, (240, 240, 240), text_rect.inflate(10, 10))
                self.screen.blit(text_surface, text_rect)
            pygame.display.flip()
            pygame.time.delay(20)

    def visualize_path(self, path):
        """
        (Obsolete) Visualize the agent path using a red line on the TUe map.
        """
        point_size = 6
        path_width = 2
        
        if self.map_surface is None:
            self._preprocess_map()
        if self.screen is None:
            pygame.init()
            self.screen = draw_TUe_map(pygame, self.screen)

        # Starting point
        start_x, start_y = self.path[0]
        pygame.draw.circle(self.screen, (239, 177, 21), (int(start_x), int(start_y)), point_size + 2)
        pygame.draw.circle(self.screen, (240, 240, 240), (int(start_x), int(start_y)), point_size)
        
        # Delivery points
        for point in self.delivery_points:
            px, py = point
            pygame.draw.circle(self.screen, (17, 167, 28), (px, py), point_size + 2)
            pygame.draw.circle(self.screen, (240, 240, 240), (px, py), point_size)
        
        # Agent path
        if len(path) > 1:
            points = [(int(point[0]), int(point[1])) for point in path]
            pygame.draw.lines(self.screen, (235, 45, 0), False, points, path_width)
        pygame.display.flip()
        
        # Keep the window open until closed by user
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.time.delay(10)
    
    def plot_map_with_path(self, path=None, delivery_points=None, figsize=(16, 10)):
        """
        Visualize the TUe map, path, and delivery points using matplotlib.
        Matplotlib is of higher resolution in comparison to pygame.
        """
        if self.map_surface is None:
            self._preprocess_map()
            
        # Get map pixels (RGB)
        map_pixels = pygame.surfarray.array3d(self.map_surface)  # (width, height, 3)
        
        # Transpose to (height, width, 3) for matplotlib
        map_pixels = np.transpose(map_pixels, (1, 0, 2))
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(map_pixels)

        if path is not None and len(path) > 1:
            px = [point[0] for point in path]
            py = [point[1] for point in path]
            ax.plot(px, py, color='red', linewidth=2, label='Path')

        if delivery_points is None:
            delivery_points = getattr(self, 'delivery_points', [])
            
        if delivery_points:
            dpx = [pt[0] for pt in delivery_points]
            dpy = [pt[1] for pt in delivery_points]
            ax.scatter(dpx, dpy, s=80, c='green', edgecolors='black', zorder=5, label='Delivery Points')

        if path is not None and len(path) > 0:
            ax.scatter([path[0][0]], [path[0][1]], s=80, c='orange', edgecolors='black', zorder=6, label='Start')
        
        building_dict = dict(zip(Buildings.building_names, Buildings.building_coors))
        for name, (bx, by) in building_dict.items():
            if not isinstance(name, float):
                ax.text(
                    bx, by, name, fontsize=10, color='white', ha='center', va='center', weight='bold',
                    path_effects=[effects.Stroke(linewidth=1, foreground='black'), effects.Normal()]
                )
        ax.axis('off')
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_aspect('equal')
        ax.set_title('TUe Map with Path and Delivery Points')
        ax.legend()
        plt.tight_layout()
        plt.show()
    

np.random.seed(123)
env = TUeMapEnv()
state, _ = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.plot_map_with_path(env.path)
