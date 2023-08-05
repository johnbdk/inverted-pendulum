"""This file contains various parameters of the environment.""" 

# external imports
import numpy as np

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