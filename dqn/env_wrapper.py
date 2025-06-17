import numpy as np
from gymnasium import spaces


class EnvWrapper:
    """
    Wraps either A1_Folder.world.Environment or TUeMapEnv such that:
      - reset() -> obs
      - step(a) -> (obs, reward, done, info)
      - it always has .action_space and .observation_space
      - seeds TUeMapEnv once
    """

    def __init__(self, env, *, is_gym_env=False, seed=None):
        """
        Args:
          env: this is either a TUeMapEnv instance or a grid-world Environment
          is_gym_env: True if `env` is already a Gymnasium Env (TUeMapEnv), otherwise False
        """
        self.env = env
        self.is_gym = is_gym_env
        self.seed = seed

        # action_space
        if hasattr(env, "action_space"):
            # TUeMapEnv or any gym.Env
            self.action_space = env.action_space
        else:
            # Grid-world with 4 discrete moves (up, down, left, right)
            self.action_space = spaces.Discrete(4)

        # observation_space
        if hasattr(env, "observation_space"):
            self.observation_space = env.observation_space
        else:
            # Grid-world state is (i,j) -> 2-dimensional vector
            # no normalization assumed
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )

    def reset(self):
        if self.is_gym:
            # we pass our stored seed into the first reset call
            obs, _ = self.env.reset(seed=self.seed)
            # then clear it so subsequent resets not reseeded
            self.seed = None
            return obs
        else:
            return self.env.reset()

    def step(self, action):
        res = self.env.step(action)
        # grid-env step (obs, reward, done) or (obs, reward, done, info)
        # gym-env step (obs, reward, term, trunc, info)
        if len(res) == 3:
            obs, reward, terminated = res
            truncated = False
            info = {}
        elif len(res) == 4:
            obs, reward, terminated, info = res
            truncated = False
        else:
            obs, reward, terminated, truncated, info = res
            #done = bool(term or trunc)
        return obs, reward, terminated, truncated, info

    def close(self):
        return getattr(self.env, "close", lambda: None)()
