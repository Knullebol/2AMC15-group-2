from argparse import ArgumentParser
import yaml
import numpy as np
import torch
from dueling_dqn.dueling_double_dqn import DuelingDoubleDQNAgent
from environment.gym import TUeMapEnv
from tqdm import trange
import matplotlib.pyplot as plt
from gymnasium import spaces
from typing import cast


def parse_args():
    p = ArgumentParser(description="DIC Dueling-Double DQN Trainer.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of episodes to train.")
    p.add_argument("--steps", type=int, default=100,
                   help="Max steps per episode.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for reproducibility.")
    p.add_argument("--logs", action="store_true",
                   help="Enable detailed logging.")
    p.add_argument("--use_distance", action="store_true",
                   help="Enable distance-based rewards.")
    p.add_argument("--use_direction", action="store_true",
                   help="Enable direction-based rewards.")
    p.add_argument("--use_stalling", action="store_true",
                   help="Enable stalling penalty.")
    return p.parse_args()


class DuelingDoubleDQNTrainingModel:
    def __init__(self, config_path='dueling_dqn/hyperparams.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)

        # hyperparameters
        self.epsilon_init = float(params.get('epsilon_init', 1.0))
        self.gamma = float(params.get('gamma', 0.99))
        self.detect_range = float(params.get('detect_range', 10.0))
        self.buffer_capacity = int(params.get('replay_memory_size', 10000))
        self.batch_size = int(params.get('mini_batch_size', 64))
        self.epsilon_min = float(params.get('epsilon_min', 0.01))
        
        self.target_update_freq = int(params.get('target_sync_freq', 1000))
        # optional DuelingDoubleDQN-specific
        self.hidden_dim = int(params.get('hidden_dim', 128))
        self.lr = float(params.get('lr', 1e-3))
        self.dueling_type = str(params.get('dueling_type', 'average'))

    def train(self, episodes, steps, seed, logs,
              use_distance, use_direction, use_stalling):
        # set device and seeds
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # initialize environment
        env = TUeMapEnv(
            detect_range=int(self.detect_range),
            goal_threshold=15.0,
            max_steps=steps,
            use_distance=use_distance,
            use_direction=use_direction,
            use_stalling=use_stalling
        )

        # derive dimensions from first observation and action space
        obs, _ = env.reset(seed=seed)
        state_dim = obs.shape[0]
        action_space = cast(spaces.Discrete, env.action_space)
        action_dim = action_space.n

        # agent initialization
        agent = DuelingDoubleDQNAgent(
            state_dim, action_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            gamma=self.gamma,
            batch_size=self.batch_size,
            buffer_capacity=self.buffer_capacity,
            target_update_freq=self.target_update_freq,
            dueling_type=self.dueling_type,
            seed=seed,
            device=device
        )

        # metrics tracking
        all_rewards = []
        best_reward = -float('inf')
        best_episode = 0
        best_path = None
        best_goal = False
        best_steps_to_goal = float('inf')
        num_goal = 0
        action_counts = [0] * action_dim
        total_steps = 0

        # epsilon decay schedule
        eps = self.epsilon_init
        decay_episodes = int(episodes * 0.8)
        eps_decay = (self.epsilon_init - self.epsilon_min) / max(1, decay_episodes)

        pbar = trange(episodes, desc='Training DuelingDoubleDQN', disable=not logs)
        for ep in pbar:
            obs, _ = env.reset()
            state = obs  # numpy array of observations
            ep_reward = 0.0

            for t in range(steps):
                # action selection
                action = agent.select_action(state, eps)
                action_counts[action] += 1

                # environment step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = next_obs
                done = terminated or truncated
                ep_reward += reward
                total_steps += 1

                # memory & learning step
                agent.push(state, action, reward, next_state, done)
                loss = agent.update()

                state = next_state
                if terminated:
                    num_goal += 1
                if done:
                    break

            # decay epsilon
            # eps = max(self.epsilon_min, eps - eps_decay)
            if ep < decay_episodes:
                eps = max(self.epsilon_min, eps - eps_decay)
            else:
                eps = self.epsilon_min

            # verbose logging
            if logs:
                pbar.set_description(
                    f"Ep {ep:3d} | R {ep_reward:5.1f} | eps {eps:.3f}"
                )

            all_rewards.append(ep_reward)
            # track best performance
            if ep_reward > best_reward:
                agent.save('best_model_dueling.pth')
                best_reward = ep_reward
                best_episode = ep
                best_path = env.path
                best_goal = terminated
                best_steps_to_goal = t if terminated else float('inf')

        # final summary
        print("Training completed.")
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        print(f"Best Ep: {best_episode}, Reward: {best_reward:.2f}, Goal: {best_goal}, Steps to goal: {best_steps_to_goal}")
        print(f"Total goals reached: {num_goal}, Avg reward: {avg_reward:.2f}")
        for i in range(action_dim):
            pct = (action_counts[i]/total_steps)*100 if total_steps > 0 else 0
            print(f"Action {i}: {action_counts[i]} times ({pct:.1f}%)")

        # visualize results
        if best_path is not None:
            env.plot_map_with_path(best_path, is_training=False)
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    trainer = DuelingDoubleDQNTrainingModel()
    trainer.train(
        episodes=args.episodes,
        steps=args.steps,
        seed=args.random_seed,
        logs=args.logs,
        use_distance=args.use_distance,
        use_direction=args.use_direction,
        use_stalling=args.use_stalling
    )