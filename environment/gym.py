from gymnasium import spaces
from TUeMap import draw_TUe_map
from TUeMap import Buildings

import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects

# Agent starting position
SPAR_LOCATION = (415, 286)
REWARD_STEP = -0.25
REWARD_OBSTACLE = -2.0
REWARD_GOAL = 250.0
MAX_ATTEMPTS = 100  # Maximum attempts to generate delivery points
DEFAULT_MAX_STEPS = 1000 # Default maximum steps for an episode

class TUeMapEnv(gym.Env):
    """
    A gymnasium continuous environment that simulates delivery tasks on the TU/e campus.
    """
    def __init__(self, goal_threshold=20.0, num_delivery_points=2, max_steps=DEFAULT_MAX_STEPS):
        """
        Initialize the TU/e Map environment.
        Args:
            goal_threshold (float): Distance threshold to consider agent reached goal (in pixels)
            num_delivery_points (int): Number of delivery points to generate
            max_steps (int): Maximum number of steps before truncation
        """
        super(TUeMapEnv, self).__init__()
        self.width, self.height = 1280, 860
        self.rng = np.random.default_rng()
        self.max_steps = max_steps
        self.steps_count = 0

        # Action: 0=forward, 1=turn left, 2=turn right
        self.action_space = spaces.Discrete(3)

        # State: agent_x, agent_y, agent_orientation, goal_x, goal_y (all normalized to [0,1])
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.screen = None

        # Cache map_surface and access_mask in __init__
        pygame.init()
        self.map_surface = draw_TUe_map(pygame, None)

        # Create access_mask
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
        self.path = []

        # Movement parameters
        self.forward_speed = 10  # pixels per step
        self.turn_speed = np.deg2rad(20)  # radians per step

        # Multiple delivery points (set in reset())
        self.delivery_points = []
        self.current_goal_idx = 0
        self.num_delivery_points = num_delivery_points

        # Minimum distance between agent start and goal (in pixels)
        self.min_goal_distance = 100.0
        self.goal_threshold = goal_threshold

    def _preprocess_map(self):
        """
        Mask buildings and roads to mark accessible areas and obstacles.
        """
        # Map surface and access mask are now cached in __init__
        # This method is kept for backward compatibility
        if self.map_surface is None:
            pygame.init()
            self.map_surface = draw_TUe_map(pygame, None)

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
        Reset the environment with the agent at the SPAR location (415, 286)
        and generate multiple delivery points on walkable tiles that are at least
        min_goal_distance pixels away from the start and from each other.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset step counter
        self.steps_count = 0

        # Set agent's start position to fixed SPAR location
        px, py = SPAR_LOCATION
        theta = self.rng.uniform(-np.pi, np.pi)
        self.state = np.array([px, py, theta], dtype=np.float32)
        start_pos = self.state[:2]

        # Generate delivery points
        self.delivery_points = self.add_delivery_points(
            start_pos=start_pos,
            n_points=self.num_delivery_points
        )

        # Sort delivery points based on distance from start position
        if self.delivery_points:
            self.delivery_points.sort(key=lambda point: np.linalg.norm(np.array(point) - start_pos))

        self.current_goal_idx = 0
        self.path = [self.state[:2].copy()]

        # Create normalized observation
        obs = self._get_normalized_obs()

        return obs, {}

    def _get_normalized_obs(self):
        """
        Creates a normalized observation vector from the current state.
        """
        x, y, theta = self.state

        # Get current goal coordinates
        if self.delivery_points and self.current_goal_idx < len(self.delivery_points):
            gx, gy = self.delivery_points[self.current_goal_idx]
        else:
            # Default to agent position if no goal is set (should not happen)
            gx, gy = x, y

        # Normalize x, y coordinates by dividing by width/height
        norm_x = x / (self.width - 1)
        norm_y = y / (self.height - 1)

        # Normalize theta to [0, 1] (from [-pi, pi])
        norm_theta = (theta + np.pi) / (2 * np.pi)

        # Normalize goal coordinates
        norm_gx = gx / (self.width - 1)
        norm_gy = gy / (self.height - 1)

        return np.array([norm_x, norm_y, norm_theta, norm_gx, norm_gy], dtype=np.float32)

    def step(self, action):
        """
        Take a step in the environment based on the action.
        Returns:
            observation (numpy.ndarray): The normalized observation after taking the action
            reward (float): The reward received for taking the action
            terminated (bool): Whether the episode has terminated (all goals reached)
            truncated (bool): Whether the episode was truncated (max steps exceeded)
        """
        x, y, theta = self.state

        # Calculate distance to current goal before action
        distance_before = float('inf')
        if self.delivery_points and self.current_goal_idx < len(self.delivery_points):
            agent_pos = self.state[:2]
            current_goal = np.array(self.delivery_points[self.current_goal_idx])
            distance_before = np.linalg.norm(agent_pos - current_goal)

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

        # Only update position if accessible
        px, py = int(nx), int(ny)
        if self.access_mask[px, py]:
            self.state = np.array([nx, ny, ntheta], dtype=np.float32)
        else:
            # If not accessible, stay in place, only update theta if turning
            if action == 0:
                # Forward blocked
                self.state = np.array([x, y, theta], dtype=np.float32)
            else:
                self.state = np.array([x, y, ntheta], dtype=np.float32)
            reward = REWARD_OBSTACLE  # Penalty for hitting obstacle

        # Calculate distance to current goal after action
        distance_after = float('inf')
        if self.delivery_points and self.current_goal_idx < len(self.delivery_points):
            agent_pos = self.state[:2]
            current_goal = np.array(self.delivery_points[self.current_goal_idx])
            distance_after = np.linalg.norm(agent_pos - current_goal)

            # Add reward proportional to the reduction in distance
            if distance_before != float('inf'):
                reward += (distance_before - distance_after) * 0.1

        # Check if agent reached the current goal
        at_current_goal = self._at_goal()
        terminated = False
        if at_current_goal:
            reward = REWARD_GOAL
            self.current_goal_idx += 1
            if self.current_goal_idx >= len(self.delivery_points):
                terminated = True

        # Increment step counter
        self.steps_count += 1

        # Check for truncation condition
        truncated = False

        # Truncate if max steps exceeded
        if self.steps_count >= self.max_steps:
            truncated = True

        self.path.append(self.state[:2].copy())

        obs = self._get_normalized_obs()

        return obs, reward, terminated, truncated, {}

    def _at_goal(self):
        """
        Checks if the agent has reached the current goal.
        """
        if not self.delivery_points or self.current_goal_idx >= len(self.delivery_points):
            return False

        agent_pos = self.state[:2]
        current_goal = np.array(self.delivery_points[self.current_goal_idx])

        # Calculate Euclidean distance between agent and current goal
        distance = np.linalg.norm(agent_pos - current_goal)

        # Return True if distance is less than threshold
        return distance <= self.goal_threshold

    def add_delivery_points(self, start_pos, points=[], n_points=2):
        """
        Add multiple delivery points to the environment.
        """
        if not points:
            points = []
            n = 0
            attempts = 0
            while n < n_points and attempts < MAX_ATTEMPTS:
                attempts += 1
                x = int(self.rng.uniform(0, 1) * (self.width-1))
                y = int(self.rng.uniform(0, 1) * (self.height-1))

                # Check if point is on accessible tile
                if not self.access_mask[x, y]:
                    continue

                # Check if point is far enough from start position
                point = np.array([x, y])
                if np.linalg.norm(start_pos - point) < self.min_goal_distance:
                    continue

                # Check if point is far enough from other delivery points
                too_close = False
                for existing_point in points:
                    if np.linalg.norm(np.array(existing_point) - point) < self.min_goal_distance:
                        too_close = True
                        break

                if not too_close:
                    points.append((x, y))
                    n += 1

        return points

    def plot_map_with_path(self, path=None, delivery_points=None, figsize=(16, 10), is_training=False):
        """
        Visualize the TUe map, path, and delivery points using matplotlib.
        Matplotlib is of higher resolution in comparison to pygame.
        """
        # Skip plotting during training to improve performance
        if is_training:
            return
        path_width = 2
        current_goal_size = 150
        remaining_goal_size = 80
        completed_goal_size = 60
        start_point_size = 100
        current_goal_border = 8
        remaining_goal_border = 5
        completed_goal_border = 4
        start_point_border = 6

        if self.map_surface is None:
            self._preprocess_map()

        # Get map pixels (RGB)
        map_pixels = pygame.surfarray.array3d(self.map_surface)  # (width, height, 3)

        # Transpose to (height, width, 3) for matplotlib
        map_pixels = np.transpose(map_pixels, (1, 0, 2))

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(map_pixels)

        # Plot agent path (red)
        if path is not None and len(path) > 1:
            px = [point[0] for point in path]
            py = [point[1] for point in path]
            ax.plot(px, py, color='red', linewidth=path_width, label='Agent Path')

        if delivery_points is None:
            delivery_points = getattr(self, 'delivery_points', [])

        # Plot delivery points
        if delivery_points:
            # Get current goal index
            current_idx = getattr(self, 'current_goal_idx', 0)

            # Plot current goal (blue star)
            if current_idx < len(delivery_points):
                current_goal = delivery_points[current_idx]
                ax.scatter([current_goal[0]], [current_goal[1]], s=current_goal_size, c='blue', marker='*',
                          edgecolors='black', zorder=current_goal_border, label='Current Goal')

            # Plot remaining delivery points (green)
            remaining_points = [pt for i, pt in enumerate(delivery_points)
                               if i != current_idx and i >= current_idx]
            if remaining_points:
                rpx = [pt[0] for pt in remaining_points]
                rpy = [pt[1] for pt in remaining_points]
                ax.scatter(rpx, rpy, s=remaining_goal_size, c='green', marker='o',
                          edgecolors='black', zorder=remaining_goal_border, label='Remaining Delivery Points')

            # Plot completed delivery points (gray)
            completed_points = [pt for i, pt in enumerate(delivery_points) if i < current_idx]
            if completed_points:
                cpx = [pt[0] for pt in completed_points]
                cpy = [pt[1] for pt in completed_points]
                ax.scatter(cpx, cpy, s=completed_goal_size, c='gray', marker='o',
                          edgecolors='black', zorder=completed_goal_border, label='Completed Delivery Points')

        # Plot start point (orange)
        if path is not None and len(path) > 0:
            ax.scatter([path[0][0]], [path[0][1]], s=start_point_size, c='orange', marker='o',
                      edgecolors='black', zorder=start_point_border, label='Start')

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
        ax.set_title('TUe Map with Path and Delivery Points')
        ax.legend()
        plt.tight_layout()
        plt.show()


def run_random_episode(env, max_steps=DEFAULT_MAX_STEPS, seed=None):
    obs, _ = env.reset(seed=seed)
    total_reward, steps = 0, 0
    done = False
    while not done and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if truncated:
            break
    return total_reward, done, steps


if __name__ == "__main__":
    env = TUeMapEnv()
    # Seed will be passed to reset() instead of using global np.random.seed

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    total_reward, success, steps = run_random_episode(env, seed=None)
    print(f"Episode finished after {steps} steps with total reward {total_reward}")
    print(f"Goal reached: {success}")
    env.plot_map_with_path(env.path, is_training=False)
