import numpy as np
from agents.base_agent import BaseAgent


class TDSarsaAgent(BaseAgent):
    """
    An on-policy (SARSA) variant of the Temporal Difference agent. Using the epsilon greedy policy.

    @param num_actions (int) : Number of actions available in the environment
    @param alpha (float) : Learning rate (0 < alpha <= 1)
    @param gamma (float) : Discount variable (0 <= gamma <= 1)
    @param epsilon (float) : Exploration rate (0 <= epsilon <= 1)
    """
    def __init__(self, num_actions: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1): 
        super().__init__

        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Intitialize Q(s, a) values table, mapping states (s) to value arrays for each action (a)
        self.Q: dict[tuple[int, int], np.ndarray] = {}

        # Intitilize last state and action for updates
        self._last_state: tuple[int, int] = None
        self._last_action: int = None

    """
    Checks if the state has already exists in the Q table, if not, intizilize it using numpy.zeros for each possible action

    @param state (tuple[int, int]) : The state to check
    """
    def _check_state_in_table(self, state: tuple[int, int]):
        if(state not in self.Q):
            self.Q[state] = np.zeros(self.num_actions, dtype = float)

    """
    Take an action based on the epislon greedy policy.
    If a random value between 0 and 1 is smaller than epsilon, pick a random action. 
    Otherwise pick the best action.

    @param state (tuple[int, int]) : State to pick the action from

    @returns action (int) : Optimal action according to the policy
    """
    def take_action(self, state: tuple[int, int]):
        # Check if state exists already
        self._check_state_in_table(state)

        # Implement the epsilon greedy policy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[state])

        # Update last values
        self._last_action = action
        self._last_state = state

        return action
    
    """
    Using the SARSA update function : Q(s, a) = Q(s, a) + alpha*( reward + gamma*Q(s', a') - Q(s, a) )
    
    @param state (tuple[int, int]) : Current state
    @param reward (float) : Reward for the transmission
    @param action (int) : current action to take
    """
    def update(self, state: tuple[int, int], reward: float, action: int):
        # If no previous state exists, cannot be updated
        if(self._last_state == None):
            return
        
        self._check_state_in_table(state)
        self._check_state_in_table(self._last_state)
        
        # Get Q(s, a) (Q_old) and Q(s', a') (Q_new)
        Q_old = self.Q[self._last_state][self._last_action]
        Q_new = self.Q[state][action]

        # Update Q(s, a)
        self.Q[self._last_state][self._last_action] = Q_old + self.alpha*(reward + self.gamma*Q_new - Q_old)