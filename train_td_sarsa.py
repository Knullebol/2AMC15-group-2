"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.td_sarsa_agent import TDSarsaAgent
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
    from agents.td_sarsa_agent import TDSarsaAgent

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


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int):
    """Main loop of the program."""
    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed) #, agent_start_pos=(1,13))

        state = env.reset()
        ACTIONS = 4
        # Initialize agent
        agent = TDSarsaAgent(num_actions = ACTIONS)
        
        # Always reset the environment to initial state
        state = env.reset()

        # Since it is on policy, choose the first action now
        action = agent.take_action(state)

        for _ in trange(iters):
            
            # Take the chosen action
            next_state, reward, terminated, info = env.step(action)

            # Choose the next action
            next_action = agent.take_action(next_state)

            # Update SARSA with the next action and state
            agent.update(next_state, reward, next_action) 

            # Go to next state and action
            state = next_state
            action = next_action        

            # If terminal, reset (new state and action)
            if terminated:
                state = env.reset()
                action = agent.take_action(state)

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed) #, agent_start_pos=(1,13))


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)