"""This file contains various parameters of the environment.""" 

# external imports
import numpy as np
import math

# number of discritized states in the environment
NVEC = 9

# Discrete Action Space (used for mappings)
DISK_ACTIONS = [-3, -1, 0, +1, +3]
DISK_NUM_ACTIONS = len(DISK_ACTIONS)

# number of required consecutive steps in the 
# target vicinity in order to end RL episode
WIN_STEPS = 100

# maximum angle error between disk and target (rads)
DESIRED_TARGET_ERROR = 0.2

# multi target angles
MULTI_TARGET_ANGLES = [
    -np.pi/18,  # 10 degrees to the left from top
    0,          # exactly at the top
    +np.pi/18]  # 10 degrees to the right from top

NORMALIZE_ANGLE = lambda angle: angle - (math.ceil((angle + math.pi)/(2*math.pi))-1)*2*math.pi

# single target reward functions
TARGET_ERROR_SINGLE = lambda theta : np.pi - np.abs(theta)
REWARD_SINGLE_THETA     = lambda theta:         1.0 * np.cos(TARGET_ERROR_SINGLE(theta))             # -1 (theta=0 (down), worst) to +1 (theta=+π or -π (up), best)
REWARD_SINGLE_OMEGA     = lambda theta, omega:  0.025 * np.abs(omega) * np.cos(theta)   # -1 (fastest when theta=+π or -π, worst) to +1 (fastest when theta=0, best)
REWARD_SINGLE_ACTION    = lambda action:        -(1/6) * np.abs(action)                 # -0.5 (highest absolute voltage, worst) to +0.5 (lowest absolute voltage, best)
REWARD_SINGLE = lambda theta, omega, action : \
    REWARD_SINGLE_THETA(theta) + \
    REWARD_SINGLE_OMEGA(theta, omega) + \
    REWARD_SINGLE_ACTION(action)

# multi-target reward functions
TARGET_ERROR_MULTI = lambda theta, target: np.pi - np.abs(np.pi - np.abs(theta - target))                                          # -1 (theta=opposite of target, worst) to +1 (theta=at target, best)
REWARD_MULTI_THETA      = lambda theta, target:         1.0 * np.cos(TARGET_ERROR_MULTI(theta, target))                      # -1 (fastest when at target, worst) to +1 (fastest when opposite of target, best)
REWARD_MULTI_OMEGA      = lambda theta, omega, target:  -0.025 * np.abs(omega) * np.cos(TARGET_ERROR_MULTI(theta, target))   # -0.5 (highest absolute voltage, worst) to +0.5 (lowest absolute voltage, best)
REWARD_MULTI_ACTION     = lambda action:        -(1/6) * np.abs(action)
REWARD_MULTI = lambda theta, omega, action, target : \
    REWARD_MULTI_THETA(theta, target) + \
    REWARD_MULTI_OMEGA(theta, omega, target) + \
    REWARD_MULTI_ACTION(action)


