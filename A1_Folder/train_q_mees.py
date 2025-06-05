from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import random
import numpy as np

try:
    from world import Environment
    from agents.random_agent import RandomAgent
except ModuleNotFoundError:
    from os import path, pardir, sys
    root_path = path.abspath(path.join(path.abspath(__file__), pardir, pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent


def parse_args():
    p = ArgumentParser(description="DIC RL Trainer (RandomAgent or Q-Learning)")
    p.add_argument("GRID", type=Path, nargs="+", help="Path(s) to .npy grid file(s)")
    p.add_argument("--no_gui", action="store_true", help="Disable GUI for speed")
    p.add_argument("--sigma", type=float, default=0.1, help="Environment stochasticity (sigma)")
    p.add_argument("--fps", type=int, default=30, help="GUI frames per second")
    p.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    p.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")
    # Q-Learning hyperparams
    p.add_argument("--alpha", type=float, default=0.1, help="Learning rate α")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor γ")
    p.add_argument("--epsilon", type=float, default=0.1, help="Initial ε for ε-greedy")
    p.add_argument("--epsilon_decay", type=float, default=1.0, help="Multiplicative ε decay")
    p.add_argument("--min_epsilon", type=float, default=0.01, help="Minimum ε floor")
    return p.parse_args()


def main(
    grid_paths, no_gui, sigma, fps,
    episodes, max_steps, random_seed,
    alpha, gamma, epsilon, epsilon_decay, min_epsilon
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    for grid in grid_paths:
        env = Environment(
            grid_fp=grid, no_gui=no_gui,
            sigma=sigma, target_fps=fps,
            random_seed=random_seed
        )

        # choose agent
        use_q = any([
            alpha != 0.1, gamma != 0.9,
            epsilon != 0.1,
            epsilon_decay != 1.0,
            min_epsilon != 0.01
        ])
        if use_q:
            from agents.q_agent_v1 import QLearningAgent
            agent = QLearningAgent(
                alpha=alpha, gamma=gamma,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                min_epsilon=min_epsilon
            )
        else:
            agent = RandomAgent()

        # training loop (episodic)
        returns = []
        successes = 0
        for ep in range(episodes):
            state = env.reset()
            if use_q:
                agent.reset()
            total_return = 0.0
            for t in range(max_steps):
                action = agent.take_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(next_state, reward, info["actual_action"])
                total_return += reward
                state = next_state
                if done:
                    successes += 1
                    break
            returns.append(total_return)

        # save learning curve
        out_dir = Path("results/")
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / f"{grid.stem}_returns.csv", "w") as f:
            f.write("episode,return\n")
            for i, r in enumerate(returns, 1):
                f.write(f"{i},{r}\n")
        print(f"[{grid.name}] Success rate: {successes}/{episodes}")

        # force greedy for evaluation
        if use_q:
            agent.epsilon = 0.0

        Environment.evaluate_agent(
            grid_fp=grid,
            agent=agent,
            max_steps=max_steps,
            sigma=sigma,
            agent_start_pos=env.agent_start_pos,
            random_seed=random_seed,
            show_images=True
        )

if __name__ == '__main__':
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.sigma,
        args.fps,
        args.episodes,
        args.max_steps,
        args.random_seed,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.epsilon_decay,
        args.min_epsilon
    )