# external imports
import numpy as np
from gym import Wrapper
from gym.spaces import MultiDiscrete, Discrete

# local imports
from rl.env.unbalanced_disk_env import UnbalancedDisk


class Discretizer(Wrapper):
    def __init__(self, env, nvec=10):
        super(Discretizer, self).__init__(env) #sets self.env
        
        if isinstance(nvec,int): #nvec in each dimention
            self.nvec = [nvec]*np.prod(env.observation_space.shape,dtype=int)
        else:
            self.nvec = nvec
        self.nvec = np.array(nvec) #(Nobs,) array
        
        self.observation_space = MultiDiscrete(self.nvec) #b)
        self.olow, self.ohigh = np.array([-np.pi, -40]), np.array([np.pi, 40])

        self.prev_theta = 0
        self.curr_theta = 0

    def discretize(self,observation): #b)
        return tuple(((observation - self.olow)/(self.ohigh - self.olow)*self.nvec).astype(int)) #b)
        
    def step(self, action):
        
        # make step in environment
        observation, reward, done, info = self.env.step(action) #b)
        observation_discrete = self.discretize(observation)

        # update observation
        self.prev_theta = self.curr_theta
        self.curr_theta = observation[0]

        # calculate reward
        curr_dist = np.pi - np.abs(self.curr_theta)
        prev_dist = np.pi - np.abs(self.prev_theta)
        # reward = 100 * (prev_dist - curr_dist)

        # reward = - ((np.pi - np.abs(self.curr_theta))**2 + 0.1 * observation[1] + 0.001 * action**2)

        # print('[ACTION      ]: voltage: [low:%d   val:%d        high:%d]' % (-3, action, +3))
        # print('[OBSERVATION ]: theta  : [low:%.1f val:%.4f [%d/%d] high:%.1f])' % (self.olow[0], observation[0], observation_discrete[0], self.nvec, self.ohigh[0]))
        # print('[OBSERVATION ]: omega  : [low:%.1f val:%.4f [%d/%d] high:%.1f]' % (self.olow[1], observation[1], observation_discrete[1], self.nvec, self.ohigh[1]))
        # print('[REWARD      ]: %.2f' % (reward))
        return observation_discrete, reward, done, info #b)

    def reset(self):
        return self.discretize(self.env.reset()) #b)


class CustomUnbalancedDisk(UnbalancedDisk):
    """
    Class to override the original UnbalancedDisk Gym Environment.
    """

    def __init__(self):
        super(CustomUnbalancedDisk, self).__init__()

        # Modify the original action space
        self.action_space = Discrete(5) #discrete

    def step(self, action):
        state, reward, done, info = super().step(action)
        return state, reward, done, info

    def reset(self):
        return super().reset()

    def render(self, mode='human'):
        return super().render(mode)
