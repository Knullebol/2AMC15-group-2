"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    # from agents.random_agent import RandomAgent
    from agents.old_mc_agents.mc_on_policy_agent import McOnPolicyAgent
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
    # from agents.random_agent import RandomAgent
    from agents.old_mc_agents.mc_on_policy_agent import McOnPolicyAgent

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
   

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)
        
        # Assuming 4 actions (up, down, left, right)
        n_actions = 4  
        
        # Initialize agent
        agent = McOnPolicyAgent(n_actions=n_actions, gamma=0.9, epsilon=0.1, max_episode_length=100)
        
        for _ in trange(iters):
            state = env.reset()
            episode_len = 0

            while episode_len < agent.max_episode_length:

                action = agent.take_action(state)

                state, reward, terminated, info = env.step(action)
                episode_len += 1 

                if terminated:
                    break
                
                agent.update(state, reward, info["actual_action"])
            agent.end_episode()

        # Evaluation step (optional for MC agents, but required by assignment)
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)