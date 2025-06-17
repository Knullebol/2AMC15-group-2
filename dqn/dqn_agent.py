from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_layer_dim, actions_layer_dim, hidden_layer_dim=128):
        super(DQN, self).__init__()

        # 2 hidden layers, no padding
        # Going from the input space, to the 2 hidden layers, to the output space
        self.fc1 = nn.Linear(state_layer_dim, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc3 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc4 = nn.Linear(hidden_layer_dim, actions_layer_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x