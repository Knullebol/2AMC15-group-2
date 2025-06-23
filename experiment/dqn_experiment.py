from experiment.base_experiment import BaseExperiment, ExperimentConfig
from dqn.dqn_agent import DQN
from dqn.experience_replay import ReplayMemory, Transition
from environment.gym import TUeMapEnv
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=64,
                 buffer_capacity=10000, target_update_freq=1000, device='cpu', seed=42):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_count = 0
        self.action_dim = action_dim

        # Networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and memory
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayMemory(buffer_capacity)

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def select_action(self, state, epsilon=0.1):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < epsilon:
            # Use stored action_dim for random action selection
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def push(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        # Note: done flag is not used in basic DQN implementation but kept for interface compatibility
        self.memory.append(state_tensor, action, next_state_tensor, reward)

    def update(self):
        """Update the network using replay memory"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        # Convert to tensors
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)

        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch).squeeze()

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path):
        """Save the model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)

    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']


class DQNExperiment(BaseExperiment):
    def create_agent(self, state_dim: int, action_dim: int):
        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=self.config.lr,
            gamma=self.config.gamma,
            batch_size=self.config.batch_size,
            buffer_capacity=self.config.memory_size,
            target_update_freq=self.config.target_sync_freq,
            device=str(self.device),
            seed=self.config.seed
        )

    def create_environment(self):
        return TUeMapEnv(
            detect_range=self.config.detect_range,
            goal_threshold=self.config.goal_threshold,
            max_steps=self.config.max_steps,
            use_distance=self.config.use_distance,
            use_direction=self.config.use_direction,
            use_stalling=self.config.use_stalling,
            destination=self.config.destination
        )


def run_dqn_experiment(
    batch_size: int = 64,
    memory_size: int = 10000,
    lr: float = 0.001,
    detect_range: int = 4,
    episodes: int = 500,
    max_steps: int = 200,
    seed: int = 42,
    name: str = "dqn",
    **kwargs
):
    config = ExperimentConfig(
        # Training parameters
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,

        # Agent parameters
        batch_size=batch_size,
        memory_size=memory_size,
        lr=lr,

        # Environment parameters
        detect_range=detect_range,

        # Experiment metadata
        name=name,
        **kwargs
    )

    experiment = DQNExperiment(config, experiment_name="DQN_Experiments")
    return experiment.run()
