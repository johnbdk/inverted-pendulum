# system imports
import math

# external imports
import numpy as np
from gym.spaces import Discrete, Box

# local imports
from rl.env.unbalanced_disk.unbalanced_disk import UnbalancedDisk
from config.env import (
    NVEC, 
    WIN_STEPS,
    DISK_ACTIONS,
    DISK_NUM_ACTIONS,
    DESIRED_TARGET_ERROR,
    MULTI_TARGET_ANGLES,
    TARGET_ERROR_SINGLE,
    TARGET_ERROR_MULTI,
    REWARD_SINGLE,
    REWARD_MULTI,
    NORMALIZE_ANGLE
)

class CustomUnbalancedDiskSingle(UnbalancedDisk):
    """
    Class to override the original UnbalancedDisk Gym Environment with a single target.
    """

    def __init__(self, action_space_type : str = 'discrete'):
        super(CustomUnbalancedDiskSingle, self).__init__()

        # Modify the original action space
        self.action_space_type = action_space_type
        if self.action_space_type == 'discrete':
            self.action_space = Discrete(DISK_NUM_ACTIONS) #discrete

        # other parameters
        self.complete_steps = 0

    def step(self, action):
        # map discrete action choice to float voltage
        if self.action_space_type == 'discrete':
            action = DISK_ACTIONS[action]

        # step action into environment
        obs, reward, done_timeout, info = super().step(action)

        # apply theta angle normalization in-place in (-pi/pi) range
        obs[0] = NORMALIZE_ANGLE(obs[0])

        # extract observation variables
        theta = obs[0]
        omega = obs[1]

        # calculate done
        target_error = TARGET_ERROR_SINGLE(theta)
        if target_error <= DESIRED_TARGET_ERROR:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # calculate reward
        reward = REWARD_SINGLE(theta=theta,
                               omega=omega,
                               action=action)

        # append stats
        info['theta'] = theta
        info['omega'] = omega
        info['theta_error'] = target_error
        info['target_dev'] = 0
        info['complete_steps'] = self.complete_steps

        return obs, reward, done, info

    def reset(self):
        self.complete_steps = 0
        return super().reset()

    def render(self, mode='human'):
        return super().render(mode)

class CustomUnbalancedDiskMulti(UnbalancedDisk):
    """
    Class to override the original UnbalancedDisk Gym Environment with multiple targets.
    """

    def __init__(self):
        super(CustomUnbalancedDiskMulti, self).__init__()

        # change observation space for multi targets
        low =   [-float('inf'), -40, -np.pi]
        high =  [+float('inf'), +40, +np.pi]
        self.observation_space = Box(low=np.array(low,dtype=np.float32),
                                     high=np.array(high,dtype=np.float32),
                                     shape=(len(low),))

        # other parameters
        self.complete_steps = 0
        self.active_target = 0

    def step(self, action):
        # step action into environment
        obs, _, done_timeout, info = super().step(action)

        # apply theta angle normalization in-place in (-pi/pi) range
        obs[0] = NORMALIZE_ANGLE(obs[0])

        # extract observation variables
        theta = obs[0]
        omega = obs[1]
        target_dev = obs[2]
        target = NORMALIZE_ANGLE(math.pi + target_dev)

        # calculate done
        target_error = TARGET_ERROR_MULTI(theta, target)
        if target_error <= DESIRED_TARGET_ERROR:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # calculate reward
        reward = REWARD_MULTI(theta=theta,
                              omega=omega,
                              action=action,
                              target=target)

        # append info stats
        info['theta'] = theta
        info['omega'] = omega
        info['complete_steps'] = self.complete_steps
        info['theta_error'] = target_error
        info['target_dev'] = target_dev

        return obs, reward, done, info
    
    def change_target(self, target_angle):
        self.active_target = target_angle
    
    def get_obs(self):
        obs = super().get_obs()
        obs = np.append(obs, self.active_target)
        return obs

    def reset(self):
        self.complete_steps = 0
        self.active_target = np.random.choice(MULTI_TARGET_ANGLES, size=1)
        return super().reset()

    def render(self, mode='human'):
        return super().render(mode)
    