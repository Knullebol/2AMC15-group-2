"""
Training script for the Value Iteration agent.
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

try:
    from world import Environment
    from agents.value_iteration_agent import ValueIterationAgent
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
    from agents.value_iteration_agent import ValueIterationAgent

def parse_args():
    p = ArgumentParser(description="Value Iteration Agent Trainer")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Max number of iterations for value iteration.")
    p.add_argument("--eval_steps", type=int, default=1000,
                   help="Number of steps for evaluation.")
    p.add_argument("--gamma", type=float, default=0.75,
                   help="Discount factor for future rewards.")
    p.add_argument("--verbose", action="store_true",
                   help="Print verbose output during training.")
    p.add_argument("--random_seed", type=int, default=42,
                   help="Random seed value for the environment.")
    return p.parse_args()

def main(args):
    """Main loop of the program."""

    for grid_path in args.GRID:
        grid_name = grid_path.stem
        print(f"\n====== Training on grid: {grid_name} ======\n")

        env = Environment(grid_path, args.no_gui, sigma=args.sigma, 
                          target_fps=args.fps, random_seed=args.random_seed)
        
        # Initialize agent
        agent = ValueIterationAgent(
            env=env, 
            grid_path=grid_path, 
            gamma=args.gamma, 
            sigma=args.sigma, 
            atol=1e-4
        )

        state = env.reset()
        
        # Train agent
        print("\nTraining Value Iteration Agent...\n")
        agent.run_value_iteration(max_iterations=args.iter, verbose=args.verbose)

        if args.verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=120)
            print("\nValue Function:")
            print(agent.V.T)
            print("\nPolicy:")
            print(agent.policy.T)

        start_pos = None
        if 'A1' in str(grid_path):
            start_pos = (3, 11)
        elif 'test' in str(grid_path):
            start_pos = (1, 13)
        elif 'large' in str(grid_path):
            start_pos = (14, 18)
        
        # Evaluate the agent
        print("\nEvaluating agent performance...\n")
        Environment.evaluate_agent(
            grid_fp=grid_path,
            agent=agent,
            max_steps=args.eval_steps,
            sigma=args.sigma,
            agent_start_pos=start_pos,
            random_seed=args.random_seed,
            show_images=not args.no_gui
        )

if __name__ == '__main__':
    args = parse_args()
    main(args)