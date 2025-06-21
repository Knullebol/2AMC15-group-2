import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)


class ReplayBuffer:
    """Simple FIFO experience replay buffer"""
    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).long().unsqueeze(1),
            torch.from_numpy(rewards).float(),
            torch.from_numpy(next_states).float(),
            torch.from_numpy(dones).float().unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, dueling_type="average"):
        super(DuelingDQN, self).__init__()
        # shared feature layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # value stream
        self.value_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_out = nn.Linear(hidden_dim // 2, 1)
        # advantage stream
        self.adv_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.adv_out = nn.Linear(hidden_dim // 2, action_dim)

        assert dueling_type in {"naive", "average", "max"}, f"Unknown dueling_type: {dueling_type}"
        self.dueling_type = dueling_type
        self.apply(orthogonal_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        features = F.relu(self.fc3(x))

        V = F.relu(self.value_fc(features))
        V = self.value_out(V)            # shape [B,1]
        A = F.relu(self.adv_fc(features))
        A = self.adv_out(A)              # shape [B,A]

        if self.dueling_type == "naive":
            Q = V + A
        elif self.dueling_type == "average":
            Q = V + (A - A.mean(dim=1, keepdim=True))
        else:  # max
            Q = V + (A - A.max(dim=1, keepdim=True)[0])
        return Q


class DuelingDoubleDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        batch_size=32,
        buffer_capacity=10000,
        target_update_freq=1000,
        dueling_type="average",
        device=None,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim, dueling_type).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim, dueling_type).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, seed)

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            n_actions = self.policy_net.adv_out.out_features
            prob_forward = 0.7
            prob_turn = (1 - prob_forward) / (n_actions - 1)
            probs = [prob_forward] + [prob_turn] * (n_actions - 1)
            action = np.random.choice(n_actions, p=probs)
            return action
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.policy_net(state)
        return int(qvals.argmax(dim=1).item())

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones.squeeze(1)) * self.gamma * next_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())