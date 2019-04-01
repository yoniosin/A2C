import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
from copy import deepcopy

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """

    def __init__(self, env, model, prioritizer, nsteps=5, gamma=0.99):
        self.n_active_envs = env.venv.n_active_envs
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.batch_ob_shape = (self.n_active_envs * nsteps,) + env.observation_space.shape
        self.ob_shape = (self.n_active_envs,) + env.observation_space.shape
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        self.active_envs = None
        self.all_env_dict = {'obs': deepcopy(self.obs),
                             'states': deepcopy(self.states),
                             'dones': deepcopy(self.dones),
                             'stackedobs': deepcopy(self.env.stackedobs)
                             }

        self.obs = None
        self.states = None
        self.dones = None
        self.env.stackedobs = None
        self.prioritizer = prioritizer
        self.prio_val = None

    def run(self):
        # overwrite super class

        # We initialize the lists that will contain the mb of experiences
        self.set_active_envs()
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, envs_activations = [], [], [], [], [], [[] for _ in range(self.nenv)]
        mb_td = []
        mb_states = self.states
        epinfos = []

        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            for i in self.active_envs:
                envs_activations[i].append(n)

            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)
            self.prio_val = values

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            obs, rewards, dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            self.states = states
            self.dones = dones
            for i, (done, env_i) in enumerate(zip(dones, self.active_envs)):
                self.all_env_dict['dones'][env_i] = done
                if done:
                    self.obs[i] = self.obs[i] * 0
                self.obs[i][:] = obs[i]
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for i, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[i] = rewards

        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_actions = mb_actions.reshape(self.batch_action_shape)
        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos

    @staticmethod
    def take_active_envs(obs, env_activation, last_step):
        obs_t = list(zip(*obs))
        mb_obs_lists = [[obs[last_step - j] for j in env_activation[i]] for i, obs in enumerate(obs_t)]
        mb_obs_active = [obs for sublist in mb_obs_lists for obs in sublist]
        return mb_obs_active

    def set_active_envs(self):
        random_env_idx = self.prioritizer.pick_active_envs(self.prio_val)
        self.active_envs = list(random_env_idx)
        self.env.venv.set_active_envs(random_env_idx)

        self.obs = [self.all_env_dict['obs'][i] for i in self.active_envs]
        self.states = self.all_env_dict['states'][self.active_envs] if self.all_env_dict['states'] else self.all_env_dict['states']
        self.dones = [self.all_env_dict['dones'][i] for i in self.active_envs]
        self.env.stackedobs =  [self.all_env_dict['stackedobs'][i] for i in self.active_envs]

