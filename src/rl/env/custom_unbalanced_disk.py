# external imports
import math
import numpy as np
from gym import Wrapper
from gym.spaces import MultiDiscrete, Discrete

# local imports
from rl.env.unbalanced_disk import UnbalancedDisk

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


class CustomUnbalancedDisk(UnbalancedDisk):
    """
    Class to override the original UnbalancedDisk Gym Environment.
    """

    def __init__(self):
        super(CustomUnbalancedDisk, self).__init__()

        # Modify the original action space
        self.action_space = Discrete(5) #discrete

        self.complete_steps = 0
        self.prev_theta = 0

    def step(self, action):
        action = [-3, -1, 0, 1, 3][action]

        obs, reward, done_timeout, info = super().step(action)

        # normalize theta angle to (-pi/pi) range
        obs[0] = obs[0] - (math.ceil((obs[0] + math.pi)/(2*math.pi))-1)*2*math.pi
        
        theta = obs[0]
        omega = obs[1]


        # calculate done
        if math.pi - abs(theta) <= 0.2:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)


        # calculate reward
        # if (math.pi - abs(theta)) <= np.pi/24:
        #     reward = 500
        # elif (math.pi - abs(theta)) <= np.pi/3:
        #     reward = 100 - ((math.pi - abs(theta))**2 + 0.01*omega**2 + 0.1*action**2)
        # else:
        #     reward = theta**2 + 0.01*omega**2 + 0.1*action**2

        # reward = - (math.pi - abs(theta))**2 - 0.01*omega**2 + 0.1*action**2
        
        r_angle = 1.0 * np.cos(np.pi - theta)                                           # -1 (theta=0 (down), worst) to +1 (theta=+π or -π (up), best)
        r_speed = 0.025 * np.abs(omega) * np.cos(theta)                                 # -1 (fastest when theta=+π or -π, worst) to +1 (fastest when theta=0, best)
        r_voltage = -(1/6)*np.abs(action)                                               # -0.5 (highest absolute voltage, worst) to +0.5 (lowest absolute voltage, best)
        reward = r_angle + r_speed + r_voltage                                          # -2.5 (worst) to +2.5 (best)
    
        # r_angle = 1.0 * np.cos(np.pi - theta)                                           # -1 (theta=0 (down), worst) to +1 (theta=+π or -π (up), best)
        # r_speed = 0.025 * np.abs(omega) * (np.cos(theta/2)**2 - np.sin(theta/2)**2)     # -1 (fastest when theta=+π or -π, worst) to +1 (fastest when theta=0, best)
        # r_voltage = (1/6) * action * np.sin(theta) * np.exp(-(omega**2))                # -0.5 (highest absolute voltage, worst) to +0.5 (lowest absolute voltage, best)
        # reward = r_angle + r_speed + r_voltage                                          # -2.5 (worst) to +2.5 (best)

        # reward = (np.cos(theta - np.pi) + 1.5)**2 - 0.25 \
        #     + 0.00125 * np.cos((theta + 1)/2)*(omega**2) \
        #     - 0.01 * action**2


        info['theta'] = theta
        info['theta_bottom'] = theta
        info['omega'] = omega
        info['complete_steps'] = self.complete_steps
        info['theta_error'] = np.pi - np.abs(info['theta'])

        self.prev_theta = theta

        return obs, reward, done, info

    def reset(self):
        self.complete_steps = 0
        return super().reset()

    def render(self, mode='human'):
        return super().render(mode)
