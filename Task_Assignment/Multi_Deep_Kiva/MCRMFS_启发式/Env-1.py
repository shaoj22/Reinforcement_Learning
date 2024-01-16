# environment class
# author: Charles Lee
# date: 2022.10.26

import numpy as np
import matplotlib.pyplot as plt
from MCRMFS_tools import Draw_Opt

class Env():
    def __init__(self, args):
        # data inside instance: T, R, col_num, row_num, N, p, s, h0, e0, g0, A
        self.instance = args["instance"]
        self.node_dim = 8 # node feature: p, s, h, e, g, assigned
        self.draw_opt = Draw_Opt(self.instance) # draw operator for render()
        self.fig = plt.figure() # fig objecti for render()

    def _transfer_state(self, p, s, h, e, g, assigned):
        state = np.zeros((self.instance.N, self.node_dim))
        for i in range(self.instance.N):
            state[i, 0] = i // self.instance.col_num
            state[i, 1] = i % self.instance.col_num
            state[i, 2] = p[i]
            state[i, 3] = s[i]
            state[i, 4] = h[i]
            state[i, 5] = e[i]
            state[i, 6] = g[i]
            state[i, 7] = assigned[i]
        return state

    def reset(self):
        """
        reset the environment and return initial state

        Return:
            state (ndarray): all information about depot and robot
            info (list[:, 2]): other informations, in this case is idxs of pickers
        """
        p = self.instance.p
        s = self.instance.s
        h = self.instance.h0
        e = self.instance.e0
        g = self.instance.g0
        assigned = [0] * self.instance.N
        state = self._transfer_state(p, s, h, e, g, assigned)
        mask = np.where(g==1)[0].tolist()
        return state, mask

    def evaluate(self, action):
        """evaluate function, return reward if apply the action

        Args:
            action (list[2]): constains rach idx and empty space idx
        Return:
            reward (double): reward if apply the action
        """
        reward = np.random.random()
        return reward

    def step(self, action):
        """
        step function, apply action to change env and reply

        Args:
            action (list[2]): constains rach idx and empty space idx
        Return:
            state (ndarray): all information about depot and robot
            reward (double): reward gain in this step
            done (int): whether the episode is done
            info (list): other informations, in this case is idxs of pickers
        """
        route_planning(action, main_robot)
        while 1:
            check_robots()
            robots_step()

        return state, reward, done, info

    def render(self):
        """
        render function to visualize solution
        information needed: p0, s0, h[t], e[t], g[t], f[t]
        """
        ax = self.fig.add_subplot(111)
        self.draw_opt.draw_t(ax, p, s, h, e, g)
        plt.draw()
        plt.pause(0.1)