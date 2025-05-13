"""
This is an agent that takes action based on value iteration.
"""
from random import randint
import numpy as np
from tqdm import trange
from world.environment import Environment
from world.grid import Grid
from world.helpers import action_to_direction
from agents import BaseAgent
from pathlib import Path


class DPAgent(BaseAgent):
    """
    A class that implements a dynamic programming agent using value iteration.
    It utilizes the Bellman Equation to build stochastic model to determine optimal actions in each cell.
    Args:
        env (Environment): The environment in which the agent operates.
        fp (Path): The path to the grid file.
        discount (float): Discount factor for future rewards.
    """
    def __init__(self, env: Environment, fp: Path, discount: float = 0.9):
        super().__init__()
        self.env = env
        self.gamma = discount
        self.grid = Grid.load_grid(fp).cells
        self.sigma = self.env.sigma   # Stochastic factor of agent action
        self.actions = [0, 1, 2, 3]  # Down, Up, Left, Right

        # Store updates
        self.V = np.zeros(self.grid.shape)      # Value function
        self.V_old = np.zeros(self.grid.shape)  # Previous value function
        self.policy = np.zeros(self.grid.shape) # Policy

    def update(self, state: tuple[int, int], value: float, action):
        """
        Update the value function and policy for the given state.
        """
        self.V[state] = value
        self.policy[state] = action

    def _prepare_next_states(self, x, y):
        """
        Get the next states (by default the adjacent 4 cells) for a given state.
        """
        next_states = []
        for action in self.actions:
            direction = action_to_direction(action)
            next_x = x + direction[0]
            next_y = y + direction[1]
            next_states.append((next_x, next_y))
        return next_states
    
    def _expected_value_of_action(self, action, next_states):
        """
        Calculate expected value for a given action.
        """
        # The action taken has a probability of 1 - sigma, while the other actions have a probability of sigma / 3
        probs = [self.sigma / 3 for _ in range(len(self.actions))]
        probs[action] = 1 - self.sigma
        q = 0
        for i, next_state in enumerate(next_states):
            reward = self.env.reward_fn(self.grid, next_state)
            # sum_{s', r} P(s', r | s, a) * (r + γV[s'])
            q += probs[i] * (reward + self.gamma * self.V[next_state])
        return q
    
    def value_iteration(self):
        """
        One iteration of Value iteration algorithm to update the value function and policy.
        """
        self.V_old = np.copy(self.V)
        for x in range(self.grid.shape[0]):  # Iterate over cells
            for y in range(self.grid.shape[1]):
                if self.grid[x, y] in [1, 2]:
                    self.V[x, y] = -100  # Penalty for wall or obstacle
                    continue
                
                action_values = np.zeros(4)
                next_states = self._prepare_next_states(x, y)

                for action in self.actions:
                    action_values[action] = self._expected_value_of_action(action, next_states)

                self.update((x, y), np.max(action_values), np.argmax(action_values))
                    

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Take action based on stored policy.
        """
        return self.policy[state]
