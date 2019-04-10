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
        '''
        obs, rews, news, infos = self.venv.step_wait()
        whole_reward = np.zeros(self.num_envs)
        whole_news = np.ones(self.num_envs, dtype=bool)
        for i in range(self.venv.num_envs):
            self.stackedobs[i] = np.roll(self.stackedobs[i], shift=-1, axis=-1)
            if i in self.venv.active_envs_set:
                idx = np.where(np.asanyarray(self.venv.active_envs_set) == i)[0]
                if news[idx]:
                    self.stackedobs[i] = 0
                self.stackedobs[i, ..., -obs.shape[-1]:] = obs[idx]
                whole_news[i] = news[idx]
                whole_reward[i] = rews[idx]
        return self.stackedobs, whole_reward, list(whole_news), infos
        '''

        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos


    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def set_active_envs(self, random_env_idx):
        self.venv.set_active_envs(random_env_idx)