from torch import nn
import torch.nn.functional as F


import torch
from torch import nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_layer_dim, actions_layer_dim, hidden_layer_dim=64):
        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, actions_layer_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine streams
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals

