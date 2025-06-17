from argparse import ArgumentParser
from dqn.dqn_agent import DQN
from dqn.experience_replay import ReplayMemory, Transition
from dqn.env_wrapper import EnvWrapper
from environment.gym import TUeMapEnv
from tqdm import trange

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of episodes to go through.")
    p.add_argument("--steps", type=int, default=100,
                   help="Number of steps per episode to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--logs", action="store_true",
                   help="Print logs of the agent")
    p.add_argument("--use_distance", action="store_true",
                   help="Allows the use of distance based rewards")
    return p.parse_args()


class DQNTrainingModel:

    def __init__(self):
        with open('dqn/hyperparams.yml', 'r') as file:
            hyperparams = yaml.safe_load(file)

        self.mini_batch_size = hyperparams['mini_batch_size']
        self.epsilon_init = hyperparams['epsilon_init']
        self.epsilon_min = hyperparams['epsilon_min']
        self.gamma = hyperparams['gamma']
        self.detect_range = hyperparams['detect_range']
        self.repMemSize = hyperparams['repMemSize']
        self.target_sync_freq = hyperparams['target_sync_freq']

    def train(self, episodes: int, steps: int, seed: int, logs: bool, use_distance: bool):

        # use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)

        raw_env = TUeMapEnv(
            detect_range=self.detect_range,
            goal_threshold=15.0,
            max_steps=steps,
            use_distance=use_distance
        )
        wrapper = EnvWrapper(raw_env, is_gym_env=True, seed=seed)
        env = wrapper

        # dynamic dimensions - no more hardcoded values
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # networks, optimizer and memory
        policy_net = DQN(state_dim, action_dim).to(device)
        target_net = DQN(state_dim, action_dim).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
        memory = ReplayMemory(self.repMemSize)

        # keeps tracks of all rewards
        all_rewards = []

        # for logging later
        best_reward = float('-inf')
        best_episode = 0
        best_path = None
        best_episode_goal_reached = False
        best_episode_goal_steps = float('inf')
        num_goal_reached = 0

        epsilon = self.epsilon_init
        epsilon_delta = (self.epsilon_init - self.epsilon_min) / (episodes - 1)

        # training loop
        tbar = trange(episodes, desc="Training DQN Agent", disable=not logs)
        for ep in tbar:
            obs = env.reset()
            state = torch.tensor(obs, dtype=torch.float32).to(device)
            total_reward = 0.0

            for t in range(steps):
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_net(state).argmax().item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = torch.tensor(next_obs, dtype=torch.float32).to(device)

                memory.append(state, action, next_state, reward)
                state = next_state
                total_reward += reward

                if len(memory) >= self.mini_batch_size:
                    batch = memory.sample(self.mini_batch_size)
                    batch = Transition(*zip(*batch))

                    S = torch.stack(batch.state).to(device)
                    A = torch.tensor(batch.action).unsqueeze(1).to(device)
                    R = torch.tensor(batch.reward, dtype=torch.float32).to(device)
                    S2 = torch.stack(batch.next_state).to(device)

                    Q = policy_net(S).gather(1, A).squeeze()
                    with torch.no_grad():
                        Q2 = target_net(S2).max(1)[0]
                    target = R + self.gamma * Q2

                    loss = F.mse_loss(Q, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # It only terminates if the agent reaches the goal
                if(terminated):
                    num_goal_reached += 1

                if(terminated or truncated):
                    break

            if ep % self.target_sync_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # When logs is selected, print all kinds of logs
            if(logs):
                tbar.set_description(
                    f"Training DQN Agent | Episode {ep:4d} | Reward: {total_reward: 6.2f} | Epsilon: {epsilon: .3f} | Device: {device} |"
                )

            # Update the best performing episode
            if(total_reward > best_reward):
                best_reward = total_reward
                best_episode = ep
                best_path = raw_env.path

                # Reached goal
                if(terminated):
                    best_episode_goal_reached = True
                    best_episode_goal_steps = t
                else:
                    best_episode_goal_reached = False
                    best_episode_goal_steps = float('inf')


            # Linear epsilon that decays to the min (per step, NOT per episode)
            epsilon -= epsilon_delta

            all_rewards.append(total_reward)

        print("Done training!")

        # summary logs
        if(logs):
            average_reward = sum(all_rewards) / episodes

            print(f"\n***********\n* Summary *\n***********\nBest episode:{best_episode: .0f} | Reward:{best_reward: 6.2f}")
            print(f"Goal reached this episode: {best_episode_goal_reached}, in{best_episode_goal_steps: .0f} steps.")
            print(f"\nTotal times goal reached:{num_goal_reached: .0f} | Average reward:{average_reward: 6.2f} ")

            # show path at the end
            raw_env.plot_map_with_path(best_path, is_training=False)

            # plot the rewards
            x = list(range(episodes))
            plt.plot(x, all_rewards)
            plt.show()


if __name__ == "__main__":
    args = parse_args()
    trainer = DQNTrainingModel()
    trainer.train(
        episodes=args.episodes,
        steps=args.steps,
        seed=args.random_seed,
        logs=args.logs,
        use_distance=args.use_distance
    )
