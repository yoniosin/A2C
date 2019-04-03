import random


class Prioritizer:
    def __init__(self, envs_num, active_envs_num):
        self.active_envs_num = active_envs_num
        self.envs_num = envs_num

    def pick_active_envs(self, _):
        return list(random.sample(list(range(self.envs_num)), self.active_envs_num))


class GreedyValuePrioritizer(Prioritizer):
    def pick_active_envs(self, prio_val):
        if prio_val is None:
            return super().pick_active_envs(prio_val)
        return (-prio_val).argsort()[:self.active_envs_num]


def PrioritizerFactory(prio_args):
    PrioritizerClass = None

    if prio_args['prio_type'] == 'random':
        PrioritizerClass = Prioritizer
    elif prio_args['prio_type'] == 'greedy':
        PrioritizerClass = GreedyValuePrioritizer

    return PrioritizerClass(prio_args['num_env'], prio_args['n_active_envs'])
