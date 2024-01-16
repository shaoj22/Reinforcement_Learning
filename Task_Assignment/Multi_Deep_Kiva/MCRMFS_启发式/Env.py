# environment class
# author: Charles Lee
# date: 2022.10.26

class Env():
    def __init__(self, instance):
        self.instance = instance

    def reset(self):
        """
        reset the environment and return initial state
        Return:
            state (ndarray): all information about depot and robot
        """
        pass

    def step(self, action):
        """
        step function, apply action to change env and reply

        Args:
            action (list[2]): constains rach idx and empty space idx
        Return:
            state (ndarray): all information about depot and robot
            reward (double): reward gain in this step
            done (int): whether the episode is done
            info (None): other informations, in this case is None
        """
        pass

    def render(self):
        """
        render function to visualize solution
        """
        # ! information needed: p0, s0, h[t], e[t], g[t], f[t]
        pass