from collections import defaultdict
import random
from agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    """
    Off-policy Q-Learning agent.

    Hyperparameters:
      alpha: learning rate
      gamma: discount factor
      epsilon: exploration probability
      epsilon_decay: multiplicative decay per step
      min_epsilon: floor for epsilon
    """
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = defaultdict(float)
        self.prev_state = None
        self.n_actions = 4

    def reset(self):
        """Clear any episode-specific state before a new episode starts."""
        self.prev_state = None

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Epsilon-greedy action selection.
        """
        # store state for update
        self.prev_state = state

        # exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # exploitation
        q_vals = [self.Q[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_vals)
        best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
        return random.choice(best_actions)

    def update(self, state: tuple[int, int], reward: float, action: int):
        """
        Q-Learning TD update.
        """
        if self.prev_state is None:
            return

        prev_s = self.prev_state
        prev_a = action
        next_s = state

        # compute max Q for next state
        max_q_next = max(
            self.Q[(next_s, a)] for a in range(self.n_actions)
        )
        current_q = self.Q[(prev_s, prev_a)]

        # TD target and error
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - current_q

        # update Q-value
        self.Q[(prev_s, prev_a)] = current_q + self.alpha * td_error

        # decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # shift prev_state
        self.prev_state = next_s