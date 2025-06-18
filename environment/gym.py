from environment.TUeMap import draw_TUe_map
from environment.TUeMap import Buildings, HEIGHT, WIDTH, ROAD_COLOR
from typing import Optional
from gymnasium import spaces

import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects


SPAR_LOCATION = (182, 60)    # Agent starting position
FORWARD_SPEED = 6            # Speed in pixels per step
DEGREE_PER_STEP = 15         # Degrees to turn per step
REWARD_STEP = -0.5           # Penalty for each step taken
REWARD_OBSTACLE = -4.0       # Penalty for hitting an obstacle
REWARD_GOAL = 250.0          # Reward for reaching the goal
DEFAULT_MAX_STEPS = 5000     # Maximum steps for an episode
REWARD_STALLING = -2         # Penalty for staying near a single spot.
STALLING_DISTANCE = 3       # Maximum distance from average last STALLING_MEMORY locations visited, to be considered stalling.
STALLING_MEMORY = 25         # Number of last visited coordinates to remember for stalling.

class TUeMapEnv(gym.Env):
    """
    A gymnasium continuous environment that simulates delivery tasks on the TU/e campus.
    """

    def __init__(self, detect_range: int,
                 goal_threshold: float,
                 max_steps: int=DEFAULT_MAX_STEPS,
                 use_distance: bool=False,
                 use_direction: bool=False,
                 use_stalling: bool=False):
        """
        Initializes the TUeMap environment.
        Args:
            detect_range (int): The agent's squared detection range.
            goal_threshold (float): Distance threshold to consider agent reached goal (in pixels).
            max_steps (int): Maximum number of steps before truncation.
        """
        super(TUeMapEnv, self).__init__()
        self.width, self.height = WIDTH, HEIGHT
        self.rng = np.random.default_rng()
        self.max_steps = max_steps
        self.steps_count = 0
        self.use_distance = use_distance
        self.use_direction = use_direction
        self.use_stalling = use_stalling

        # Action: 0=forward, 1=turn left, 2=turn right
        self.action_space = spaces.Discrete(3)

        # State: agent_x, agent_y, angle, sensor area
        self.detect_range = detect_range
        self.observation_space = spaces.Box(
            low=self.make_feature_from_sensor(detect_range=self.detect_range, mode='lower'),
            high=self.make_feature_from_sensor(detect_range=self.detect_range, mode='upper'),
        )
        self.screen = None

        self.recent_location_memory = [SPAR_LOCATION]
        # Cache map_surface and access_mask in __init__
        self.map_surface = draw_TUe_map(pygame, None)

        # Create access_mask
        map_pixels = pygame.surfarray.array3d(self.map_surface)  # shape: (width, height, 3)
        map_pixels = map_pixels.transpose(2, 0, 1)  # shape: (3, width, height)

        road_colors = [ROAD_COLOR, (180, 180, 180), (120, 120, 120)]
        mask = np.zeros((self.width, self.height), dtype=bool)

        for color in road_colors:
            color_arr = np.array(color)[:, None, None]
            matches = np.all(map_pixels == color_arr, axis=0)
            mask = np.logical_or(mask, matches)

        self.access_mask = mask.astype(np.uint8)

        # Movement parameters
        self.forward_speed = FORWARD_SPEED  # pixels per step
        self.turn_speed = np.deg2rad(DEGREE_PER_STEP)  # radians per step

        # Single delivery point (set in reset())
        self.delivery_point = (66, 123)

        # Minimum distance between agent start and goal (in pixels)
        self.min_goal_distance = 100.0
        self.goal_threshold = goal_threshold

        # Store an agent's path
        self.path = []

    def _preprocess_map(self):
        """
        Mask buildings and roads to mark accessible areas and obstacles.
        """
        # Map surface and access mask are now cached in __init__
        # This method is kept for backward compatibility
        if self.map_surface is None:
            self.map_surface = draw_TUe_map(pygame, None)

            map_pixels = pygame.surfarray.array3d(self.map_surface)  # shape: (width, height, 3)
            map_pixels = map_pixels.transpose(2, 0, 1)  # shape: (3, width, height)

            road_colors = [ROAD_COLOR, (180, 180, 180), (120, 120, 120)]
            mask = np.zeros((self.width, self.height), dtype=bool)

            for color in road_colors:
                color_arr = np.array(color)[:, None, None]
                matches = np.all(map_pixels == color_arr, axis=0)
                mask = np.logical_or(mask, matches)

            self.access_mask = mask.astype(np.uint8)

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment with the agent at the SPAR location
        and generate a single delivery point on a walkable tile that is at least
        min_goal_distance pixels away from the start.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset step counter
        self.steps_count = 0
        self.recent_location_memory = [SPAR_LOCATION]
        # Set agent's start position to fixed SPAR location
        px, py = SPAR_LOCATION
        self.state = np.array([px, py, -np.pi], dtype=np.float32)
        self.path = [self.state[:2].copy()]

        # Create observation based on state, angle and sensoring area
        obs = self.make_feature_from_sensor(self.detect_range)

        return obs, {}

    def make_feature_from_sensor(self, detect_range: int, mode: Optional[str] = None):
        """
        Create an observation feature from the agent's state and sensor area.
        
        Args:
            detect_range (int): The agent's squared detection range.
            mode (str, optional): switch between 'lower', 'upper' or None to
            return different sensor feature.
            None returns the masked array around the current position.
        """
        len_features = (2 * detect_range + 1) ** 2  # Size of the sensor area
        if mode == 'lower':
            # state dim = length of (x, y, theta) + (sensor area)
            lowerbound = np.zeros(3 + len_features, dtype=np.float32)
            return lowerbound

        elif mode == 'upper':
            upperbound = np.full(3 + len_features, 1, dtype=np.float32)
            return upperbound

        else:
            grid_len = 2 * detect_range + 1
            local_grid = np.zeros((grid_len, grid_len), dtype=np.float32)    
            x, y, theta = self.state
            x_min, x_max = int(x) - detect_range, int(x) + detect_range + 1
            y_min, y_max = int(y) - detect_range, int(y) + detect_range + 1
            
            map_x_min = max(x_min, 0)
            map_x_max = min(x_max, self.width)
            map_y_min = max(y_min, 0)
            map_y_max = min(y_max, self.height)

            grid_x_min = map_x_min - x_min
            grid_x_max = grid_x_min + (map_x_max - map_x_min)
            grid_y_min = map_y_min - y_min
            grid_y_max = grid_y_min + (map_y_max - map_y_min)
            
            local_grid[grid_x_min:grid_x_max, grid_y_min:grid_y_max] = \
                self.access_mask[map_x_min:map_x_max, map_y_min:map_y_max]
            
            normalized_x = x / (self.width - 1)
            normalized_y = y / (self.height - 1)
            normalized_theta = (theta + np.pi) / (2 * np.pi)

            # Combine normalized position and sensor feature into a single observation
            obs_feature = np.hstack(
                (normalized_x, normalized_y,
                 normalized_theta, local_grid.flatten())
            )
            
            return obs_feature
    
    def step(self, action: int):
        """
        Take a step in the environment based on the action.
        Returns:
            observation (numpy.ndarray): The normalized observation after taking the action
            reward (float): The reward received for taking the action
            terminated (bool): Whether the episode has terminated (all goals reached)
            truncated (bool): Whether the episode was truncated (max steps exceeded)
        """
        x, y, theta = self.state

        # Calculate distance to goal before action
        agent_pos = self.state[:2]
        goal = np.array(self.delivery_point)
        distance_before = np.linalg.norm(agent_pos - goal)

        # Compute angle to goal before action
        vector_agent_before = np.array([np.cos(theta), np.sin(theta)]) # Unit vector
        vector_goal_before = goal - agent_pos
        length_vector = np.linalg.norm(vector_goal_before)
        vector_goal_before = vector_goal_before / length_vector # Make unit vector

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

        # Default reward is time penalty per step
        reward = REWARD_STEP

        # ====== Penalty for hitting obstacles ======
        # Only update position if accessible
        px, py = int(nx), int(ny)
        if self.access_mask[px, py]:
            # Detect if the agent crosses white area
            can_move_forward = True
            if action == 0:
                num_samples = int(np.hypot(nx - x, ny - y)) * 2
                for i in range(1, num_samples + 1):
                    sample_x = int(x + (nx - x) * i / num_samples)
                    sample_y = int(y + (ny - y) * i / num_samples)
                    if not self.access_mask[sample_x, sample_y]:
                        can_move_forward = False
                        reward = REWARD_OBSTACLE
                        break
                    
            if not can_move_forward:
                nx, ny = x, y
            self.state = np.array([nx, ny, ntheta], dtype=np.float32)
        else:
            # If not accessible, stay in place, only update theta if turning
            if action == 0:
                # Forward blocked
                self.state = np.array([x, y, theta], dtype=np.float32)
            else:
                self.state = np.array([x, y, ntheta], dtype=np.float32)
            reward = REWARD_OBSTACLE  # Penalty for hitting obstacle

        # ====== Distance-based reward ======
        # Calculate distance to goal after action
        # Only if set to True
        if(self.use_distance):
            agent_pos = self.state[:2]
            current_goal = np.array(self.delivery_point)
            distance_after = np.linalg.norm(agent_pos - current_goal)
            reward += (distance_before - distance_after) * 1.0

        # ====== Direction-based reward ======
        # Calculate agent angle to goal (using vectors)
        # Only if set to True
        if(self.use_direction):
            agent_pos = self.state[:2]
            vector_agent_after = np.array([np.cos(ntheta), np.sin(ntheta)]) # Unit vector
            vector_goal_after = goal - agent_pos
            length_vector = np.linalg.norm(vector_goal_after)
            vector_goal_after = vector_goal_after / length_vector # Make unit vector

            # Compute the actual angles (and use clipping to make sure it does not result in NaN's)
            angle_before = np.arccos(np.clip(np.dot(vector_agent_before, vector_goal_before), -1.0, 1.0))
            angle_after = np.arccos(np.clip(np.dot(vector_agent_after, vector_goal_after), -1.0, 1.0))
            reward += (angle_before - angle_after) * 1.0

        # ====== Penalty for staying in place too long ======
        

        # ====== Reward for reaching the goal ======
        # Check if agent reached the goal
        at_goal = self._at_goal()
        terminated = False
        if at_goal:
            reward = REWARD_GOAL
            terminated = True

        # Increment step counter, terminate if max steps exceeded
        self.steps_count += 1
        truncated = False
        if self.steps_count >= self.max_steps:
            truncated = True

        self.path.append(self.state[:2].copy())
        obs = self.make_feature_from_sensor(self.detect_range)

        #===== Stalling ======
        if self.use_stalling:
            newX = self.state[0]
            newY = self.state[1]
            #Append the new location to memory
            self.recent_location_memory.append((newX, newY))
            #Calculate the average point across all remembered last-visited points.
            averageVisited = np.mean(self.recent_location_memory, axis=0)
            distanceFromAverage = np.sqrt((averageVisited[0] - newX)**2 + (averageVisited[1] - newY)**2)
            #If our new position is not far away enough from the average point, punish the agent.
            if (distanceFromAverage <= STALLING_DISTANCE):
                reward += REWARD_STALLING
            
            #Only keep track of STALLING_MEMORY number of previous positions.
            if (len(self.recent_location_memory) >= STALLING_MEMORY):
                self.recent_location_memory.pop(0) 

        return obs, reward, terminated, truncated, {}

    def _at_goal(self):
        """
        Checks if the agent has reached the goal.
        """
        agent_pos = self.state[:2]
        goal = np.array(self.delivery_point)
        distance = np.linalg.norm(agent_pos - goal)
        return distance <= self.goal_threshold

    def plot_map_with_path(self, path: Optional[list] = None, is_training: bool = False):
        """
        Visualize the TUe map, path, and delivery point using matplotlib.
        Matplotlib is of higher resolution in comparison to pygame.
        """
        # Skip plotting during training to improve performance
        if is_training:
            return
        
        path_width = 2
        point_size = 100
        point_border = 6

        if self.map_surface is None:
            self._preprocess_map()

        # Get map pixels (RGB)
        map_pixels = pygame.surfarray.array3d(self.map_surface)  # (width, height, 3)

        # Transpose to (height, width, 3) for matplotlib
        map_pixels = np.transpose(map_pixels, (1, 0, 2))

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(map_pixels)

        # Plot agent path (red)
        if path is not None and len(path) > 1:
            px = [point[0] for point in path]
            py = [point[1] for point in path]
            ax.plot(px, py, color='red', linewidth=path_width, label='Agent Path')

        # Plot the delivery point
        if self.delivery_point is not None:
            # Current goal (green)
            if self._at_goal():
                ax.scatter(
                    [self.delivery_point[0]], [self.delivery_point[1]],
                    s=point_size, c='green', marker='o',
                    edgecolors='black', linewidths=1,
                    zorder=point_border, label='Delivery point (Reached)'
                )
            else:
                # Remaining goal (blue)
                ax.scatter(
                    [self.delivery_point[0]], [self.delivery_point[1]],
                    s=point_size, c='blue', marker='o',
                    edgecolors='black', linewidths=1,
                    zorder=point_border, label='Delivery point'
                )
                # Draw a transparent circle around the delivery point
                circle = plt.Circle(
                    self.delivery_point, self.goal_threshold,
                    color='blue', alpha=0.2, fill=True, zorder=point_border
                )
                ax.add_artist(circle)
                
        # Plot start point (orange)
        if path is not None and len(path) > 0:
            ax.scatter([path[0][0]], [path[0][1]], s=point_size, c='orange', marker='o',
                       edgecolors='black', zorder=point_border, label='Start')

        building_coors_dict = Buildings.building_coors
        for name, (bx, by) in building_coors_dict.items():
            if not isinstance(name, float):
                ax.text(
                    bx, by, name, fontsize=10, color='white', ha='center', va='center', weight='bold',
                    path_effects=[effects.Stroke(linewidth=1, foreground='black'), effects.Normal()]
                )
        ax.axis('off')
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_aspect('equal')
        ax.set_title('Data Intelligence Challenge - TU/e SPAR Delivery')
        ax.legend()
        plt.tight_layout()
        plt.show()

