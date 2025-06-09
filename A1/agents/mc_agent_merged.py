from collections import defaultdict
import random
import numpy as np
from agents.base_agent import BaseAgent


class MonteCarloOnPolicyAgent(BaseAgent):
    def __init__(self, n_actions, gamma=0.9, epsilon=0.9):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.starting_epsilon = epsilon

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
                visited.add((s,a))
            else:
                self.returns[(s,a)][-1] = G
                
        for s,a in self.returns.keys():
            self.Q[s][a] = np.mean(self.returns[(s,a)])
        
        self.episode.clear()
