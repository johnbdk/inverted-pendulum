from gym.envs.classic_control import PendulumEnv
from gym.spaces import Discrete
import math
import numpy as np

WIN_STEPS = 100

class CustomPendulum(PendulumEnv):
    """
    Class to override the original UnbalancedDisk Gym Environment.
    """

    def __init__(self):
        super(CustomPendulum, self).__init__()

        # Modify the original action space
        self.action_space = Discrete(5) #discrete
        
        # extra class attributes
        self.complete_steps = 0


    def step(self, action):
        action = [[-2], [-1], [0], [1], [2]][action]

        obs, reward, done_timeout, info = super().step(action)
        
        x = obs[0]
        y = obs[1]
        omega = obs[2]
        theta = np.arccos(x)

        # calculate done
        if abs(theta) <= 0.2:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # save obs to info
        info['theta'] = theta
        info['theta_bottom'] = np.pi - theta
        info['omega'] = omega
        info['complete_steps'] = self.complete_steps
        info['theta_error'] = np.abs(theta)

        return obs, reward, done, info

    def reset(self):
        self.complete_steps = 0
        return super().reset()

    def render(self, mode='human'):
        return super().render(mode)
