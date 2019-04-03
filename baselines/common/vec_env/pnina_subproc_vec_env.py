from .subproc_vec_env import _flatten_obs
from .subproc_vec_env import *
import  random


class ModifiedSubprocVecEnv(SubprocVecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, n_active_envs, spaces=None):
        super().__init__(env_fns, spaces)
        nenvs = len(env_fns)
        self.active_envs_set = list(random.sample(range(nenvs), n_active_envs))
        self.n_active_envs = n_active_envs

    def set_active_envs(self, active_idxs):
        self.active_envs_set = active_idxs

    def step_async(self, actions):
        self._assert_not_closed()
        for action, env_i in zip(actions, self.active_envs_set):
            remote = self.remotes[env_i]
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for i, remote in enumerate(self.remotes) if i in self.active_envs_set]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for i, remote in enumerate(self.remotes):
                if i in self.active_envs_set:
                    remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
