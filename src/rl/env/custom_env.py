# external imports
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
        self.max_complete_steps = 0

    def discretize(self,observation): #b)
        return tuple(((observation - self.olow)/(self.ohigh - self.olow)*self.nvec).astype(int)) #b)
        
    def step(self, action):

        # make step in environment
        observation, _, done_timeout, info = self.env.step(action) #b)\
        
        # normalize theta angle to (-pi/pi) range
        observation[0] = observation[0] % 2*np.pi
        if observation[0] > np.pi:
            observation[0] -= 2*np.pi
        
        theta = observation[0]
        omega = observation[1]


        # discretize theta angle
        observation_discrete = self.discretize(observation)

        # calculate done
        if np.pi - np.abs(observation[0]) <= 0.1:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        self.max_complete_steps = max(self.complete_steps, self.max_complete_steps)
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # calculate reward
        reward = 0

        # encourage angles close to target
        reward += (1 - np.cos(theta))
        # reward += -(np.pi - abs(theta))**2

        # encourage large swings (reward high angular velocities
        # in the direction of the disk's current position)
        if np.sign(theta) == np.sign(omega):
            reward += 0.1 * abs(omega)
        else:
            reward -= 0.1 * abs(omega)
        
        # encourage small swings when close to target
        if abs(theta) > 0.8 * np.pi:
            reward = -0.5 * abs(omega)


        info['observation'] = observation
        info['max_complete_steps'] = self.max_complete_steps
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
