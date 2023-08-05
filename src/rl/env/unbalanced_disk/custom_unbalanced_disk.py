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
    MULTI_TARGET_ANGLES
)

@staticmethod
def normalize_angle(angle):
    """
    Normalizes the given angle to the range [-pi, pi].

    This method takes an angle and normalizes it to the interval [-pi, pi], meaning that
    if the angle is greater than pi or less than -pi, it will be "wrapped" around
    to fall within this range.

    Parameters:
        angle (float): The angle in radians that needs to be normalized.

    Returns:
        float: The normalized angle in the range [-pi, pi].
    """
    return angle - (math.ceil((angle + math.pi)/(2*math.pi))-1)*2*math.pi


class CustomUnbalancedDiskSingle(UnbalancedDisk):
    """
    Class to override the original UnbalancedDisk Gym Environment with a single target.
    """

    def __init__(self, 
                 action_space_type='discrete'):
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
        theta = normalize_angle(obs[0])
        omega = obs[1]

        # calculate done
        if math.pi - abs(theta) <= DESIRED_TARGET_ERROR:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # calculate reward
        r_angle = 1.0 * np.cos(np.pi - theta)                                           # -1 (theta=0 (down), worst) to +1 (theta=+π or -π (up), best)
        r_speed = 0.025 * np.abs(omega) * np.cos(theta)                                 # -1 (fastest when theta=+π or -π, worst) to +1 (fastest when theta=0, best)
        r_voltage = -(1/6)*np.abs(action)                                               # -0.5 (highest absolute voltage, worst) to +0.5 (lowest absolute voltage, best)
        reward = r_angle + r_speed + r_voltage                                          # -2.5 (worst) to +2.5 (best)

        # append stats
        info['theta'] = theta
        info['theta_bottom'] = theta
        info['omega'] = omega
        info['complete_steps'] = self.complete_steps
        info['theta_error'] = np.pi - np.abs(info['theta'])

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
        theta = normalize_angle(obs[0])
        omega = obs[1]
        target_dev = obs[2]

        # calculate done
        if math.pi + target_dev - abs(theta) <= DESIRED_TARGET_ERROR:
            self.complete_steps += 1
        else:
            self.complete_steps = 0
        done = done_timeout or (self.complete_steps >= WIN_STEPS)

        # calculate reward
        r_angle = 1.0 * np.cos(np.pi + target_dev - theta)                                           # -1 (theta=0 (down), worst) to +1 (theta=+π or -π (up), best)
        r_speed = 0.025 * np.abs(omega) * np.cos(theta)                                 # -1 (fastest when theta=+π or -π, worst) to +1 (fastest when theta=0, best)
        r_voltage = -(1/6)*np.abs(action)                                               # -0.5 (highest absolute voltage, worst) to +0.5 (lowest absolute voltage, best)
        reward = r_angle + r_speed + r_voltage                                          # -2.5 (worst) to +2.5 (best)

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
        self.active_target = np.random.choice(MULTI_TARGET_ANGLES, size=1)
        return super().reset()

    def render(self, mode='human'):
        return super().render(mode)