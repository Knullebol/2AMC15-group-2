"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.q_learning_agent import QLearningAgent
except ModuleNotFoundError:
    # fix up the path if being run as a script
    import sys
    import os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root not in sys.path:
        sys.path.append(root)
    from world import Environment
    from agents.q_learning_agent import QLearningAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma for environment stochasticity.")
    p.add_argument("--fps", type=int, default=30,
                   help="Render FPS (ignored if --no_gui).")
    p.add_argument("--iter", type=int, default=1000,
                   help="Total training *steps* (not episodes).")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for reproducibility.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int):
    """Main loop of the program."""
    for grid in grid_paths:

        env = Environment(grid, no_gui=no_gui, sigma=sigma, target_fps=fps, random_seed=random_seed)

        agent = QLearningAgent(num_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)

        state = env.reset()
        for step in trange(iters, desc=f"Training on {grid.name}"):
            # choose action ε-greedily
            action = agent.take_action(state)

            next_state, reward, terminated, info = env.step(action)

            # update Q-table using the actually executed action
            actual_action = info.get("actual_action", action)
            agent.update(next_state, reward, actual_action)

            state = next_state

            # when we hit terminal, reset to start a new episode
            if terminated:
                state = env.reset()

        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)
