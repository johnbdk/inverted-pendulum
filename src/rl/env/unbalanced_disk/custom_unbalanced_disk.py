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

        # normalize theta angle to (-pi/pi) range
        obs[0] = NORMALIZE_ANGLE(obs[0])
        # obs[0] = obs[0] - (math.ceil((obs[0] + math.pi)/(2*math.pi))-1)*2*math.pi
        theta = obs[0]
        omega = obs[1]

        # calculate done
        # if TARGET_ERROR_SINGLE(theta) <= DESIRED_TARGET_ERROR:
        if (np.pi - np.abs(theta)) <= DESIRED_TARGET_ERROR:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # calculate reward
        # reward = REWARD_SINGLE(theta=theta,
        #                        omega=omega,
        #                        action=action)
        reward_theta = 1.0 * np.cos(np.pi - theta)
        reward_omega = 0.025 * np.abs(omega) * np.cos(theta)
        reward_action = -(1/6) * np.abs(action)
        reward = reward_theta + reward_omega + reward_action

        # append stats
        info['theta'] = theta
        info['theta_bottom'] = theta
        info['omega'] = omega
        info['complete_steps'] = self.complete_steps
        info['theta_error'] = np.pi - np.abs(info['theta'])
        info['target_dev'] = 0

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

        # normalize theta angle to (-pi/pi) range
        # theta = NORMALIZE_ANGLE(obs[0])
        # obs[0] = obs[0] - (math.ceil((obs[0] + math.pi)/(2*math.pi))-1)*2*math.pi
        obs[0] = NORMALIZE_ANGLE(obs[0])
        theta = obs[0]
        omega = obs[1]
        target_dev = obs[2]
        # target = NORMALIZE_ANGLE(math.pi + target_dev)
        # target = (math.pi + obs[2]) - (math.ceil(((math.pi + obs[2]) + math.pi)/(2*math.pi))-1)*2*math.pi
        target = NORMALIZE_ANGLE(math.pi + obs[2])

        # calculate done
        # if TARGET_ERROR_MULTI(theta, target) <= DESIRED_TARGET_ERROR:
        _target_error_multi = np.pi - np.abs(np.pi - np.abs(theta - target))
        if _target_error_multi <= DESIRED_TARGET_ERROR:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # calculate reward
        # reward = REWARD_MULTI(theta=theta,
        #                       omega=omega,
        #                       action=action,
        #                       target=target)
        reward_multi_theta = 1.0 * np.cos(_target_error_multi)
        reward_multi_omega =  -0.025 * np.abs(omega) * np.cos(_target_error_multi)
        reward_multi_action = -(1/6) * np.abs(action)
        reward = reward_multi_theta + reward_multi_omega + reward_multi_action

        # append info stats
        info['theta'] = theta
        info['theta_bottom'] = theta
        info['omega'] = omega
        info['complete_steps'] = self.complete_steps
        info['theta_error'] = np.pi + target_dev - np.abs(info['theta'])

        return obs, reward, done, info
    
    def get_obs(self):
        obs = super().get_obs()
        obs = np.append(obs, self.active_target)
        return obs

    def reset(self):
        self.complete_steps = 0
        # self.active_target = np.random.choice(MULTI_TARGET_ANGLES, size=1)
        self.active_target = 0
        return super().reset()

    def render(self, mode='human'):
        return super().render(mode)
    