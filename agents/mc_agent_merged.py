from collections import defaultdict
import random
import numpy as np
from agents.base_agent import BaseAgent


class MonteCarloOnPolicyAgent(BaseAgent):
    def __init__(self, n_actions, gamma=0.9, epsilon=0.5, eps_min=0.01, eps_decay=0.995):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.returns = defaultdict(list)
        self.episode = []  # stores (state, action, reward) for current episode

    def take_action(self, state):
        """
        Selects action using ε-soft policy.
        """
        # ε-soft (greedy + uniform)
        best = np.argmax(self.Q[state])
        probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        probs[best] += 1 - self.epsilon
        return random.choices(range(self.n_actions), weights=probs, k=1)[0]

    def update(self, state: tuple[int, int], reward: float, action: int):
        """
        Store transition (for use at episode end).
        """
        self.episode.append((state, action, reward))

    def end_episode(self):
        """
        Performs the Monte Carlo policy update at end of episode.
        """
        G = 0
        visited = set()
        for t in reversed(range(len(self.episode))):
            s, a, r = self.episode[t]
            G = self.gamma * G + r
            if (s,a) not in visited:
                self.returns[(s,a)].append(G)
                self.Q[s][a] = np.mean(self.returns[(s,a)])
                visited.add((s,a))
        self.episode.clear()
        # decaying epsilon
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
