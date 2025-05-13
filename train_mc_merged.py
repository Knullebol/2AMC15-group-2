# train_mc_hybrid.py

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
from world import Environment
from agents.mc_agent_merged import MonteCarloOnPolicyAgent

try:
    from world import Environment
    # from agents.random_agent import RandomAgent
    from agents.mc_agent_merged import MonteCarloOnPolicyAgent
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
    from agents.mc_agent_merged import MonteCarloOnPolicyAgent

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
    """
    Main loop of the program.
    """

    # Assuming 4 actions (up, down, left, right)
    n_actions = 4  

    for grid in grid_paths:
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, random_seed=random_seed, agent_start_pos=(1,13))
        # Initialize agent
        agent = MonteCarloOnPolicyAgent(n_actions=n_actions, gamma=1, epsilon=0.9)

        for episode in trange(iters):
            state = env.reset()
            terminated = False
            steps = 0
            max_steps = 1000

            while not terminated and steps < max_steps:
                # Agent takes an action based on the latest observation and info.
                action = agent.take_action(state)

                # The action is performed in the environment
                next_state, reward, terminated, info = env.step(action)

                if terminated:
                    break

                agent.update(state, reward, info["actual_action"])
                state = next_state
                steps += 1

            agent.end_episode()
            if episode > iters*0.5:
                agent.epsilon = agent.starting_epsilon / (episode - iters*0.5)
        
        #Simulate final run with only-optimal-moves policy (Epsilon = 0)
        agent.epsilon = 0
        state = env.reset()
        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed, agent_start_pos=(1,13))

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)