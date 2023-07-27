import numpy as np


class LQ(object):
    """ 最短队列算法 """

    def __init__(self, n_states, n_actions, cfg):
        self.device = cfg.device
        self.n_states = n_states
        self.n_actions = n_actions

    def choose_action(self, state):
        """ 选择动作 """
        min_index = state.argmin()
        action = np.zeros(self.n_actions)
        action[min_index] = 1
        return action
