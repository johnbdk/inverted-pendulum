# external imports
import math
import numpy as np
from gym import Wrapper
from gym.spaces import MultiDiscrete, Discrete

# local imports
from rl.env.unbalanced_disk_env import UnbalancedDisk

WIN_STEPS = 100

class Discretizer(Wrapper):
    def __init__(self, env, nvec=10):
        super(Discretizer, self).__init__(env) #sets self.env
        
        if isinstance(nvec,int): #nvec in each dimension
            self.nvec = np.array([nvec]*np.prod(env.observation_space.shape,dtype=int)) # [nvec, nvec]
        else:
            self.nvec = np.array(nvec)
        
        print(self.nvec) # [nvec, nvec]
        
        self.observation_space = MultiDiscrete(self.nvec) #b)
        self.olow, self.ohigh = np.array([-np.pi, -40]), np.array([np.pi, 40])

        self.prev_theta = 0
        self.curr_theta = 0

        self.complete_steps = 0

    def discretize(self,observation): #b)
        return tuple(((observation - self.olow)/(self.ohigh - self.olow)*self.nvec).astype(int)) #b)
        
    def step(self, action):

        # make step in environment
        observation, _, done_timeout, info = self.env.step(action) #b)\

        # normalize theta angle to (-pi/pi) range
        observation[0] = observation[0] - (math.ceil((observation[0] + math.pi)/(2*math.pi))-1)*2*math.pi
        
        theta = observation[0]
        omega = observation[1]

        # discretize theta angle
        observation_discrete = self.discretize(observation)

        # calculate done
        if math.pi - abs(theta) <= 0.2:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # calculate reward
        if (math.pi - abs(theta)) <= np.pi/3:
            reward = 250 - ((math.pi - abs(theta))**2 + 0.1*omega**2 + 0.001*action**2)
        else:
            reward = theta**2 + 0.1*omega**2 + 0.001*action**2
        

        info['observation'] = observation
        info['complete_steps'] = self.complete_steps
        info['nvec'] = self.nvec[0]
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
