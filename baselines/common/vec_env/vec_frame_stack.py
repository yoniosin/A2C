from . import VecEnvWrapper
import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        self.stackedobs = None
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i, env_i in enumerate(self.venv.active_envs_set):
            self.stackedobs[env_i] = np.roll(self.stackedobs[env_i], shift=-1, axis=-1)
            if news[i]:
                self.stackedobs[env_i] = 0
            self.stackedobs[env_i, ..., -obs.shape[-1]:] = obs[i]
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs
