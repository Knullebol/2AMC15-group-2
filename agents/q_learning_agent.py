import numpy as np
from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent with epsilon-greedy exploration.

    Parameters
    ----------
    num_actions : int
        Number of discrete actions available in the environment.
    alpha : float
        Learning rate (0 < alpha <= 1).
    gamma : float
        Discount factor (0 <= gamma <= 1).
    epsilon : float
        Exploration rate for epsilon-greedy (0 <= epsilon <= 1).
    """

    def __init__(self, num_actions: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        super().__init__()
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table as a dict mapping state tuples to Q-value arrays
        self.q_table: dict[tuple[int, int], np.ndarray] = {}
        # store last state (before transition) for updates
        self._last_state: tuple[int, int] | None = None

    def _get_q_values(self, state: tuple[int, int]) -> np.ndarray:
        """Return Q-values for a given state, initializing to zeros if unseen."""
        if state not in self.q_table:
            # initialize Q-values to zero for all actions
            self.q_table[state] = np.zeros(self.num_actions, dtype=float)
        return self.q_table[state]

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Select an action using epsilon-greedy policy based on current Q-values.

        Args:
            state: Current state as (row, col).
        Returns:
            An integer in [0, num_actions) representing the chosen action.
        """
        # Store current state for the upcoming update
        self._last_state = state

        q_values = self._get_q_values(state)
        if np.random.rand() < self.epsilon:
            # explore
            action = np.random.randint(self.num_actions)
        else:
            # exploit: choose action with highest Q-value (break ties randomly)
            max_q = q_values.max()
            best_actions = np.flatnonzero(q_values == max_q)
            action = np.random.choice(best_actions)

        return action

    def update(self, state: tuple[int, int], reward: float, action: int):
        """
        Update Q-table based on observed transition.

        Args:
            state: New state after taking `action` (next state).
            reward: Reward received for the transition.
            action: Action taken in the previous state.
        """
        # If we have no previous state recorded, cannot update
        if self._last_state is None:
            return

        prev_state = self._last_state
        prev_action = action

        # Q(s, a)
        old_q = self._get_q_values(prev_state)[prev_action]
        # max_a' Q(s', a')
        next_max = self._get_q_values(state).max()
        # Q-learning update
        td_target = reward + self.gamma * next_max
        td_delta = td_target - old_q
        self.q_table[prev_state][prev_action] = old_q + self.alpha * td_delta
