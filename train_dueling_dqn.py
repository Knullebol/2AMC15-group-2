import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
import numpy as np
from argparse import ArgumentParser
# from dqn.dqn_agent import DQN
from dueling_dqn.dueling_dqn_agent import DuelingDQN
from dqn.experience_replay import ReplayMemory, Transition
from dqn.env_wrapper import EnvWrapper
from environment.gym import TUeMapEnv


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of episodes to go through.")
    p.add_argument("--steps", type=int, default=100,
                   help="Number of steps per episode to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


class DQNTrainingModel:

    def __init__(self):
        with open('dqn/hyperparams.yml', 'r') as file:
            hyperparams = yaml.safe_load(file)

        self.mini_batch_size = hyperparams['mini_batch_size']
        self.epsilon_init = hyperparams['epsilon_init']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.epsilon_min = hyperparams['epsilon_min']
        self.gamma = hyperparams['gamma']
        self.repMemSize = hyperparams['repMemSize']
        self.target_sync_freq = hyperparams['target_sync_freq']

    def train(self, episodes: int, steps: int, seed: int):

        torch.manual_seed(seed)
        np.random.seed(seed)

        raw_env = TUeMapEnv(
            goal_threshold=10.0,
            num_delivery_points=2,
            max_steps=steps
        )
        wrapper = EnvWrapper(raw_env, is_gym_env=True, seed=seed)
        env = wrapper

        # dynamic dimensions - no more hardcoded values
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # networks, optimizer and memory
        # policy_net = DQN(state_dim, action_dim)
        # target_net = DQN(state_dim, action_dim)
        policy_net = DuelingDQN(state_dim, action_dim)
        target_net = DuelingDQN(state_dim, action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
        memory = ReplayMemory(self.repMemSize)
        epsilon = self.epsilon_init

        # training loop
        for ep in range(episodes):
            print("GOT HERE 3")
            obs = env.reset()
            state = torch.tensor(obs, dtype=torch.float32)
            total_reward = 0.0

            for t in range(steps):
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # action = policy_net(state).argmax().item()
                        action = policy_net(state.unsqueeze(0)).argmax().item()


                next_obs, reward, done, _ = env.step(action)
                next_state = torch.tensor(next_obs, dtype=torch.float32)

                memory.append(state, action, next_state, reward)
                state = next_state
                total_reward += reward

                if len(memory) >= self.mini_batch_size:
                    batch = memory.sample(self.mini_batch_size)
                    batch = Transition(*zip(*batch))

                    S = torch.stack(batch.state)
                    A = torch.tensor(batch.action).unsqueeze(1)
                    R = torch.tensor(batch.reward, dtype=torch.float32)
                    S2 = torch.stack(batch.next_state)

                    Q = policy_net(S).gather(1, A).squeeze()
                    with torch.no_grad():
                        Q2 = target_net(S2).max(1)[0]
                    target = R + self.gamma * Q2

                    loss = F.mse_loss(Q, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                    break

            if ep % self.target_sync_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

            print(f"Episode {ep:4d} | Reward: {total_reward: 6.2f} | Epsilon: {epsilon: .3f}")


if __name__ == "__main__":
    args = parse_args()
    trainer = DQNTrainingModel()
    trainer.train(
        episodes=args.episodes,
        steps=args.steps,
        seed=args.random_seed,
    )
