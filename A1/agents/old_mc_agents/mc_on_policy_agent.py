
from collections import defaultdict
import random

import numpy as np
from agents.base_agent import BaseAgent


class McOnPolicyAgent(BaseAgent):
    """
    Monte Carlo On-Policy Agent.
    This agent uses the Monte Carlo method to learn the value of the policy it is following.
    It updates the value function based on the returns from episodes.
    """
    def __init__(self, n_actions: int, gamma: int, epsilon: int, max_episode_length: int):
        super().__init__()

        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episode_length = max_episode_length

        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.returns = defaultdict(list)
        self.policy = defaultdict(lambda: np.ones(n_actions) / n_actions)

        self.episode = []  # stores (state, action, reward) for current episode

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Selects action using ε-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return int(np.argmax(self.Q[state]))

    def update(self, state: tuple[int, int], reward: float, action: int):
        """Store transition (for use at episode end)."""
        self.episode.append((state, action, reward))

    def end_episode(self):
        """
        Performs the Monte Carlo policy update at end of episode.
        """
        G = 0
        visited = set()
        for t in reversed(range(len(self.episode))):
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward
            if (state, action) not in visited:
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])

                # Improve ε-greedy policy
                best_action = np.argmax(self.Q[state])
                self.policy[state] = np.ones(self.n_actions) * self.epsilon / self.n_actions
                self.policy[state][best_action] += 1 - self.epsilon

                visited.add((state, action))

        self.episode.clear()
