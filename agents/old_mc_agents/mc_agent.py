from random import randint
import numpy as np
import random
from collections import defaultdict

from agents import BaseAgent


class MonteCarloAgent(BaseAgent):
    
    def __init__(self, gridX: int, gridY: int, no_actions: int, epsilon: int = 0.5):
        super().__init__()
        self.gridX = gridX
        self.gridY = gridY
        self.Q = defaultdict(lambda: np.zeros(no_actions))
        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.no_actions = no_actions
    
    
    # Updates Q-Function using the average of all discounted reward per (state, action) pair.
    def update(self, state_action_reward):
        for state_action_pair in state_action_reward:
            state = state_action_pair[0]
            action = state_action_pair[1]
            self.Q[state][action] = np.average(state_action_reward[state_action_pair])


    def take_action(self, state: tuple[int, int]) -> int:
        return self.epsilon_soft(state)
    
    #Provides an action using the epsilon-soft tactic.
    def epsilon_soft(self, state):
        bestActionIndex = np.argmax(self.Q[state])
        p = []
        for i in range(self.no_actions):
            actionProb = 0
            if i == bestActionIndex:
                actionProb = 1 - self.epsilon + (self.epsilon / self.no_actions)
            else:
                actionProb = self.epsilon / self.no_actions
            p.append(actionProb)
        return random.choices([i for i in range(self.no_actions)], weights=p, k=1)[0]
    
    #DEPRECATED: Provides an action using epsilon-greedy tactic.
    def epsilon_greedy(self, state: tuple[int, int]) -> int:
        
        if random.random() <= self.epsilon:
            return random.randint(0, self.no_actions-1)
        else:
            return np.argmax(self.Q[state])