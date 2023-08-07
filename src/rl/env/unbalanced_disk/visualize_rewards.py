# external imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# system imports
import sys
import os

# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config.env import (
    DISK_ACTIONS,
    REWARD_SINGLE_THETA,
    REWARD_SINGLE_OMEGA,
    REWARD_SINGLE_ACTION,
    REWARD_SINGLE,
    REWARD_MULTI_THETA,
    REWARD_MULTI_OMEGA,
    REWARD_MULTI_ACTION,
    REWARD_MULTI
)

RESOLUTION = 100

# ---------------- S I N G L E   T A R G E T   R E W A R D ----------------

# Define ranges for theta
theta_range = np.linspace(-np.pi, np.pi, RESOLUTION)
omega_range = np.linspace(-40, 40, RESOLUTION)
action_range = np.array(DISK_ACTIONS)

# Create meshgrid for theta
theta_mesh = np.meshgrid(theta_range)
theta_mesh, omega_mesh = np.meshgrid(theta_range, omega_range)
action_mesh = np.meshgrid(action_range)


# Fixed values for omega and action
omega = 0.0
action = 0.0

# Calculate rewards
reward_single_theta = REWARD_SINGLE_THETA(theta_mesh)
reward_single_omega = REWARD_SINGLE_OMEGA(theta_mesh, omega_mesh)
reward_single_action = REWARD_SINGLE_ACTION(action_mesh)
reward_single_total = REWARD_SINGLE(theta_mesh, omega, action)

# Create a figure
fig = plt.figure(figsize=(15, 10))

# Plot REWARD_SINGLE_THETA
ax1 = fig.add_subplot(221)
ax1.plot(theta_range, reward_single_theta[0], label='REWARD_SINGLE_THETA')
ax1.legend()
ax1.set_xlabel('Theta')
ax1.set_ylabel('Reward')
ax1.grid(True)

# Plot REWARD_SINGLE_OMEGA as a 3D surface plot
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(theta_mesh, omega_mesh, reward_single_omega, cmap='viridis')
ax2.set_title('REWARD_SINGLE_OMEGA')
ax2.set_xlabel('Theta')
ax2.set_ylabel('Omega')
ax2.set_zlabel('Reward')

# Plot REWARD_SINGLE_ACTION
ax3 = fig.add_subplot(223)
ax3.plot(action_mesh, reward_single_action[0], label='REWARD_SINGLE_ACTION')
ax3.legend()
ax3.set_xlabel('Action')
ax3.set_ylabel('Reward')
ax3.grid(True)

# Plot REWARD_SINGLE
ax4 = fig.add_subplot(224)
ax4.plot(theta_range, reward_single_total[0], label='REWARD_SINGLE')
ax4.legend()
ax4.set_xlabel('Theta')
ax4.set_ylabel('Reward')
ax4.grid(True)

plt.show()
