import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from dqn.dqn_agent import DQN
from world import Environment
from dqn.experience_replay import ReplayMemory, Transition

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of episodes to go through.")
    p.add_argument("--steps", type=int, default=100,
                   help="Number of steps per episode to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--print_episodes", action='store_true',
                   help="Print information each episode")
    return p.parse_args()

class DQNTrainingModel:

    def __init__(self):
        with open('dqn/hyperparams.yml', 'r') as file:
            hyperparams = yaml.safe_load(file)

        self.replay_memory_size = hyperparams['replay_memory_size']
        self.mini_batch_size = hyperparams['mini_batch_size']
        self.epsilon_init = hyperparams['epsilon_init']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.epsilon_min = hyperparams['epsilon_min']
        self.gamma = hyperparams['gamma']
        self.repMemSize = hyperparams['repMemSize']

        self.num_state = 2
        self.num_actions = 4

    def train(self, grid_paths: list[Path], no_gui: bool, sigma: float, fps: int,
         episodes: int, steps: int, random_seed: int, print_episodes: bool):
        
        for grid in grid_paths:
        
            env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                            random_seed=random_seed, agent_start_pos=(13, 13))

            rewards_per_episode = []

            model = DQN(self.num_state, self.num_actions)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            target_model = DQN(self.num_state, self.num_actions)
            target_model.load_state_dict(model.state_dict())
            target_model.eval()

            replayMem = ReplayMemory(self.repMemSize)  # Static for now

            epsilon = self.epsilon_init

            # Set to 100 right now
            for episode in range(episodes):
                state = env.reset()
                state = torch.tensor(state, dtype = torch.float32) # Need to convert to tensors for pytorch

                episode_reward = 0.0

                # 1000 steps to reach the end goal
                for step in range(steps):

                    # Get action from agent, using the epsilon greedy policy
                    if(np.random.rand() < epsilon):
                        # Explores
                        action = np.random.randint(self.num_actions)
                    else:
                        # Exploits
                        with torch.no_grad():
                            action = torch.argmax(model(state)).item()

                    next_state, reward, terminated, info = env.step(action)
                    next_state = torch.tensor(next_state, dtype = torch.float)
                    
                    replayMem.append(state, action, next_state, reward)

                    state = next_state
                    episode_reward += reward

                    if(len(replayMem) >= self.mini_batch_size):
                        batch = replayMem.sample(self.mini_batch_size)
                        batch = Transition(*zip(*batch))

                        state_batch = torch.stack(batch.state)
                        action_batch = torch.tensor(batch.action).unsqueeze(1)
                        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
                        next_state_batch = torch.stack(batch.next_state)

                        q_vals = model(state_batch).gather(1, action_batch)
                        with torch.no_grad():
                            next_q = target_model(next_state_batch).max(1)[0]
                        target_q = reward_batch + self.gamma * next_q

                        loss = F.mse_loss(q_vals.squeeze(), target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if(terminated):
                        break

                rewards_per_episode.append(episode_reward)

                # Sync target network
                if(episode % 20 == 0):
                    target_model.load_state_dict(model.state_dict())

                # Decay epsilon (multiply times decay, NOT linear)
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if(print_episodes):
                    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.5f}")


if __name__ == '__main__':
    # Run using:
    # python "train_dqn.py" ".\grid_configs\test_grid.npy" --steps=1000 --episodes 100 --no_gui --sigma=0 --random_seed=0 --print_episodes
    args = parse_args()

    model = DQNTrainingModel()
    model.train(args.GRID, args.no_gui, args.sigma, args.fps, args.episodes, args.steps, args.random_seed, args.print_episodes)