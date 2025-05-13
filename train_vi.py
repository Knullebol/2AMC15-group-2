"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

try:
    from world import Environment
    from agents.dp_agent import DPAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.dp_agent import DPAgent

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
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


def train_agent(agent, v_iter=3000, print_cell_value=False):
    np.set_printoptions(precision=5, linewidth=300, suppress=True)
    print("\nTraining agent with value iteration.\n")
    for _ in trange(v_iter):
        agent.value_iteration()
        if np.allclose(agent.V, agent.V_old, atol=1e-5):
            print(f'Converged after {_ + 1} iterations.')
            break
        if _ == v_iter - 1 and print_cell_value:
            print('\n',agent.V.T)
            print(agent.policy.T)
    return agent
    

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)
        
        # Initialize agent
        agent = DPAgent(env, fp=grid, discount=0.9)
        
        # Always reset the environment to initial state
        state = env.reset()
        agent = train_agent(agent, v_iter=1000, print_cell_value=False)

        # Evaluate the agent
        start = (3, 11) if 'A1' in str(grid) else None
        Environment.evaluate_agent(grid, agent, iters, sigma, agent_start_pos=start, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)