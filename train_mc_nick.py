"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.mc_agent import MonteCarloAgent
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
    from agents.random_agent import RandomAgent

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
                          random_seed=random_seed)

        state = env.reset()
        ACTIONS = 4
        DISCOUNT = 1
        # Initialize agent
        agent = MonteCarloAgent(env.grid.shape[0], env.grid.shape[1], no_actions=ACTIONS, epsilon=0.9)
        # Always reset the environment to initial state
        state = env.reset()
        state_action_rewards = {}
        for episode in trange(iters):

            #Decaying-Epsilon. Encourages exploration with a high epsilon during the first half
            # of our iterations, and decays it afterwards to enforce exploitation for finding an optimal path.
            if (episode > iters*0.5):
                agent.epsilon = agent.starting_epsilon / (episode - iters*0.5)

            state = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            terminated = False
            episode_iters = 0
            
            #Simulate full episode
            while not terminated and episode_iters < 5_000:

                episode_states.append(state)
                action = agent.take_action(state)
                state, reward, terminated, info = env.step(action)

                episode_actions.append(info["actual_action"])
                episode_rewards.append(reward)
                episode_iters += 1
                
            visited = []
            G = 0
            #Iterate over episode in reverse
            for t in reversed(range(len(episode_states))):
                currState = episode_states[t]
                currAction = episode_actions[t]
                currReward = episode_rewards[t]
                
                G = G*DISCOUNT + currReward
                
                state_action_pair = (currState, currAction)
                
                #Mark the last occurrence of (state,action) pair and note its discounted reward.
                if state_action_pair not in visited:
                    visited.append(state_action_pair)
                    if state_action_pair not in state_action_rewards:
                        state_action_rewards[state_action_pair] = []
                    state_action_rewards[state_action_pair].append(G)
                else:
                    state_action_rewards[state_action_pair][-1] = G
            #At the end of every episode, update the agent using the average of
            # all discounted rewards accumulated thus far.
            agent.update(state_action_rewards)

        #Simulate final run with only-optimal-moves policy (Epsilon = 0)
        state = env.reset()
        agent.epsilon = 0
        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)