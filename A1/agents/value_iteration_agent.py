"""
Value Iteration agent using Dynamic Programming.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from tqdm import trange

from agents import BaseAgent
from world.environment import Environment
from world.grid import Grid
from world.helpers import action_to_direction


class ValueIterationAgent(BaseAgent):
    def __init__(
        self,
        env: Environment,
        grid_path: Path,
        gamma: float = 0.75,
        sigma: float | None = None,
        atol: float = 1e-4, # convergence threshold
    ):
        super().__init__()
        self.env = env
        self.grid = Grid.load_grid(grid_path).cells
        self.rows, self.cols = self.grid.shape

        self.gamma = gamma
        self.sigma = sigma if sigma is not None else env.sigma
        self.atol = atol

        self.actions = [0, 1, 2, 3]

        # value & policy tables
        self.V = np.zeros((self.rows, self.cols), dtype=float)
        self.policy = np.full((self.rows, self.cols), -1, dtype=int)

        self._precompute_rewards()

    def update(self, state: tuple[int, int], reward: float, action: int):
        pass

    def _precompute_rewards(self):
        self.rewards = np.zeros((self.rows, self.cols), dtype=float)
        for i, j in np.ndindex(self.rows, self.cols):
            self.rewards[i, j] = self.env.reward_fn(self.grid, (i, j))

    @lru_cache(maxsize=None)
    def get_next_state(self, x: int, y: int, action: int) -> tuple[int, int]:
        """
        Cached transition: applies action_to_direction, checks bounds & walls.
        """
        dx, dy = action_to_direction(action)
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.rows and 0 <= ny < self.cols:
            if self.grid[nx, ny] not in (1, 2):
                return nx, ny
        return x, y

    def compute_q_values(self, x: int, y: int) -> np.ndarray:
        """
        Returns an array [Q(s,0), Q(s,1), Q(s,2), Q(s,3)].
        If this is a terminal goal cell, returns zeros.
        """
        if self.grid[x, y] == 3:
            return np.zeros(len(self.actions), dtype=float)

        q = np.zeros(len(self.actions), dtype=float)
        # distribution of unintended moves:
        base_p = self.sigma / (len(self.actions) - 1)
        for a in self.actions:
            # assign correct probability
            probs = np.full(len(self.actions), base_p)
            probs[a] = 1.0 - self.sigma
            # accumulate expectation
            for ra, p in enumerate(probs):
                nx, ny = self.get_next_state(x, y, ra)
                q[a] += p * (self.rewards[nx, ny] + self.gamma * self.V[nx, ny])
        return q

    def _update_value_and_policy(self) -> float:
        """
        Single sweep: updates self.V and self.policy in-place;
        returns max delta for convergence checking.
        """
        max_delta = 0.0
        for i, j in np.ndindex(self.rows, self.cols):
            cell = self.grid[i, j]
            # obstacle or wall
            if cell in (1, 2):
                # keep V (likely negative) and no-action
                self.V[i, j] = self.rewards[i, j]
                self.policy[i, j] = -1
                continue

            q_vals = self.compute_q_values(i, j)
            best_a = int(np.argmax(q_vals))
            best_v = q_vals[best_a]

            delta = abs(self.V[i, j] - best_v)
            if delta > max_delta:
                max_delta = delta

            self.V[i, j] = best_v
            self.policy[i, j] = best_a

        return max_delta

    def run_value_iteration(self, max_iterations: int = 1000, verbose: bool = False):
        """
        Repeatedly sweeps until delta < atol or iteration limit.
        """
        for it in trange(max_iterations, desc="Value Iteration"):
            delta = self._update_value_and_policy()
            if verbose:
                print(f"Iter {it+1:4d}  delta={delta:.6f}")
            if delta < self.atol:
                if verbose:
                    print(f"Converged after {it+1} iterations.")
                break
        else:
            if verbose:
                print("Max iterations reached; may not have fully converged.")

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Returns the policy action for a state.
        If no valid action, defaults to 0.
        """
        x, y = state
        a = self.policy[x, y]
        return int(a) if a >= 0 else 0