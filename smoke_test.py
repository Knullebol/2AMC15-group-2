import sys
from pathlib import Path

# 1) Compute your project root directory (where smoke_test.py lives)
ROOT = Path(__file__).resolve().parent

# 2) Insert ROOT on sys.path so top-level imports work
sys.path.insert(0, str(ROOT))

# 3) Insert A1_Folder on sys.path so `import world.*` resolves to A1_Folder/world
sys.path.insert(1, str(ROOT / "A1_Folder"))

# 4) Insert environment on sys.path so `import environment.*` resolves
sys.path.insert(2, str(ROOT / "environment"))

import numpy as np
from dqn.env_wrapper import EnvWrapper

# Now these imports will succeed:
from world.environment   import Environment
from environment.gym     import TUeMapEnv


def test_grid(grid_path):
    grid_path = Path(grid_path)
    print("=== GRID-WORLD SMOKE TEST ===")
    env = EnvWrapper(Environment(grid_path),
                     is_gym_env=False)   # discrete grid
    obs = env.reset()
    print("Reset observation:", obs)
    print("Obs space:", env.observation_space)
    print("Act space:", env.action_space)
    sample_action = env.action_space.sample()
    step_out = env.step(sample_action)
    print(f"Step({sample_action}) →", step_out)
    print()

def test_map(seed=42):
    print("=== TU/e MAP SMOKE TEST ===")

    # Two wrappers with the same seed
    env1 = EnvWrapper(TUeMapEnv(), is_gym_env=True, seed=seed)
    env2 = EnvWrapper(TUeMapEnv(), is_gym_env=True, seed=seed)

    # 1) Reset both
    obs1 = env1.reset()
    obs2 = env2.reset()

    # 2) Draw exactly one action *once*
    action = env1.action_space.sample()

    # 3) Step both with that same action
    nxt1 = env1.step(action)[0]
    nxt2 = env2.step(action)[0]

    # 4) Report
    print("Reset1:", obs1)
    print("Reset2:", obs2)
    print("Same reset? ", np.allclose(obs1, obs2))

    print(f"Action chosen for both: {action}")
    print("Step1:", nxt1)
    print("Step2:", nxt2)
    print("Same step?  ", np.allclose(nxt1, nxt2))

    print("Obs space:", env1.observation_space)
    print("Act space:", env1.action_space)
    print()

if __name__ == "__main__":
    # Replace with your actual grid file path
    GRID_PATH = "A1_Folder/grid_configs/test_grid.npy"
    test_grid(GRID_PATH)
    test_map(seed=123)
