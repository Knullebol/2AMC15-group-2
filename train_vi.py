"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

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
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount factor.")
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
        if np.allclose(agent.V, agent.V_old, atol=1e-16):
            print(f'Converged after {_ + 1} iterations.')
            break
        if _ == v_iter - 1 and print_cell_value:
            print('\n',agent.V.T)
            print(agent.policy.T)
    return agent


def plot_values(agent, path=None, grid_name=None):
    """
    Plot the values of the grid.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title("Value Function and Policy")

    # Create a copy of the value grid
    value_grid = agent.V.copy()

    # Mask wall cells (set them to NaN)
    value_grid[agent.grid == 1] = np.nan  # Assuming walls are marked as 1
    value_grid[agent.grid == 2] = np.nan  # If there are other wall markers

    # Plot the heatmap
    cax = ax.matshow(value_grid.T, cmap=cm.viridis, interpolation='none',
                     extent=[-0.5, value_grid.shape[1] - 0.5, 
                             value_grid.shape[0] - 0.5, -0.5])

    # Overlay black cells for walls
    for i in range(agent.grid.shape[0]):
        for j in range(agent.grid.shape[1]):
            if agent.grid[i, j] in [1, 2]:  # Wall markers
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color='black'))

    # Add arrows to represent actions
    action_to_arrow = {
        0: (0, 0.25),  # Down
        1: (0, -0.25),   # Up
        2: (-0.25, 0),  # Left
        3: (0.25, 0)    # Right
    }
    for i in range(agent.grid.shape[0]):
        for j in range(agent.grid.shape[1]):
            if agent.grid[i, j] not in [1, 2, 3]:  # Skip walls and targets
                action = int(agent.policy[i, j])
                if action in action_to_arrow:
                    dx, dy = action_to_arrow[action]
                    ax.arrow(i, j, dx, dy, head_width=0.2, head_length=0.2, fc='white', ec='white')

    
    # Add gridlines for gaps
    ax.set_xticks(np.arange(-0.5, value_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, value_grid.shape[0], 1), minor=True)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(which="major", size=0)  # Hide major ticks
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)  # Hide minor ticks
    

    # Add title and save the plot
    plt.savefig(path + f"/value_function_{grid_name}.png")
    


def plot_recorded_values(agent, path=None, grid_name=None):
    """
    Plot the recorded values of the agent.
    """
    fig, ax = plt.subplots()
    ax.plot(agent.recorded_values)
    ax.set_title("Recorded Values")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(f"Value at {agent._first_available_cell()}")
    plt.savefig(path + f"/recorded_values_{grid_name}.png")
    

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, gamma: float, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)
        
        # Initialize agent
        agent = DPAgent(env, fp=grid, discount=gamma)
        
        # Always reset the environment to initial state
        state = env.reset()
        agent = train_agent(agent, v_iter=1000, print_cell_value=False)

        # Plot the values
        save_path = "D:/TUe/Quartile_4/Data Intelligence/figures"
        plot_values(agent, save_path, grid_name=grid.stem)
        plot_recorded_values(agent, save_path, grid_name=grid.stem)
        
        # Evaluate the agent
        start = None
        if 'A1' in str(grid):
            start = (3, 11)
        elif 'test' in str(grid):
            start = (1, 13)
        elif 'large' in str(grid):
            start = (1, 1)
        Environment.evaluate_agent(grid, agent, iters, sigma, agent_start_pos=start, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.gamma, args.random_seed)