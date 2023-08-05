# external imports
import numpy as np
from gym import Wrapper
from gym.spaces import MultiDiscrete, Discrete

# local imports
from config.env import (
    NVEC, 
)

class Discretizer(Wrapper):
    def __init__(self, env, nvec=NVEC):
        super(Discretizer, self).__init__(env) #sets self.env
        
        if isinstance(nvec,int): #nvec in each dimension
            self.nvec = np.array([nvec]*np.prod(env.observation_space.shape,dtype=int)) # [nvec, nvec]
        else:
            self.nvec = np.array(nvec)
        
        print(self.nvec) # [nvec, nvec]
        
        self.observation_space = MultiDiscrete(self.nvec) #b)
        self.olow = env.observation_space.low
        self.ohigh = env.observation_space.high

        # bound inf values
        self.olow[self.olow == -float('inf')] = -np.pi
        self.ohigh[self.ohigh == float('inf')] = np.pi 


    def discretize(self,observation): #b)
        return tuple(((observation - self.olow)/(self.ohigh - self.olow)*self.nvec).astype(int)) #b)
        
    def step(self, action):

        # make step in environment
        observation, reward, done, info = self.env.step(action) #b)\

        # discretize theta angle
        observation_discrete = self.discretize(observation)

        info['nvec'] = self.nvec[0]
        return observation_discrete, reward, done, info #b)

    def reset(self):
        return self.discretize(self.env.reset()) #b)

