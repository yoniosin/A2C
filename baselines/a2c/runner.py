import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
import random


class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.n_active_envs = env.venv.n_active_envs
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.batch_ob_shape = (self.n_active_envs * nsteps,) + env.observation_space.shape
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        self.active_envs = None

    def run(self):
        # overwrite super class

        # We initialize the lists that will contain the mb of experiences
        # mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, envs_activations = [[] for _ in range(self.nenv)], [[] for _ in range(self.nenv)], [[] for _ in range(self.nenv)], [[] for _ in range(self.nenv)], [[] for _ in range(self.nenv)], [[] for _ in range(self.nenv)]
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, envs_activations = [], [], [], [], [], [[] for _ in range(self.nenv)]
        mb_states = self.states
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            self.set_active_envs()
            active_obs = np.take(self.obs, self.active_envs, axis=0)
            active_states = np.take(self.states, self.active_envs) if self.states is not None else None
            active_dones = np.take(self.dones, self.active_envs)

            actions, values, states, _ = self.model.step(active_obs, S=active_states, M=active_dones)

            # Append the experiences
            mb_obs.append(np.copy(active_obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(active_dones)

            # Take actions in env and look the results
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            for i, (done, env_i) in enumerate(zip(dones, self.active_envs)):
                envs_activations[env_i].append(n)
                self.dones[env_i] = done
                if done:
                    self.obs[env_i] = self.obs[env_i] * 0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(active_dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

    def set_active_envs(self):
        random_env_idx = set(random.sample(list(range(self.env.venv.num_envs)), self.n_active_envs))
        self.active_envs = list(random_env_idx)
        self.env.venv.set_active_envs(random_env_idx)
