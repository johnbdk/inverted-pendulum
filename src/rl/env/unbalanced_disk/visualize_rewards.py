# external imports
import matplotlib.pyplot as plt
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
    REWARD_MULTI,
    MULTI_TARGET_ANGLES,
    NORMALIZE_ANGLE
)

RESOLUTION = 100

# ---------------- S I N G L E   T A R G E T   R E W A R D ----------------

# Create a figure
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Single Target Rewards', fontsize=16)


# Define ranges for all values
theta_range = np.linspace(-np.pi, np.pi, RESOLUTION)
omega_range = np.linspace(-40, 40, RESOLUTION)
action_range = np.array(DISK_ACTIONS)

# Fixed values for omega and action
theta_fixed_bottom = 0.0
theta_fixed_top = np.pi
omega_fixed_stopped = 0.0
omega_fixed_full_speed = 0.0
action_fixed_stop = 0.0
action_fixed_full_throttle = DISK_ACTIONS[-1]


# Plot REWARD_SINGLE_THETA
data = REWARD_SINGLE_THETA(theta_range)
ax1 = fig.add_subplot(231)
ax1.plot(theta_range, data)
ax1.set_title('Theta Reward')
ax1.set_xlabel('Theta')
ax1.set_ylabel('Reward')
ax1.grid(True)

# Plot REWARD_SINGLE_OMEGA as a 3D surface plot
theta_mesh, omega_mesh = np.meshgrid(theta_range, omega_range)
data = REWARD_SINGLE_OMEGA(theta_mesh, omega_mesh)
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(theta_mesh, omega_mesh, data, cmap='viridis')
ax2.set_title('Omega Reward')
ax2.set_xlabel('Theta')
ax2.set_ylabel('Omega')
ax2.set_zlabel('Reward')

# Plot REWARD_SINGLE_ACTION
ax3 = fig.add_subplot(233)
data = REWARD_SINGLE_ACTION(action_range)
c = data
ax3.plot(action_range, data)
ax3.set_title('Action Reward')
ax3.set_xlabel('Action')
ax3.set_ylabel('Reward')
ax3.grid(True)

# Plot Total Reward as a 3D Surface plot (theta-omega, action=0)
theta_mesh, omega_mesh = np.meshgrid(theta_range, omega_range)
data = REWARD_SINGLE(theta_mesh, omega_mesh, action_fixed_stop)
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot_surface(theta_mesh, omega_mesh, data, cmap='viridis')
ax4.set_title('TOTAL REWARD (a=0)')
ax4.set_xlabel('Theta')
ax4.set_ylabel('Omega')
ax4.set_zlabel('Reward')

# Plot Total Reward as a 3D Surface plot (omega-action, theta=0)
omega_mesh, action_mesh = np.meshgrid(omega_mesh, action_range)
data = REWARD_SINGLE(theta_fixed_bottom, omega_mesh, action_mesh)
ax5 = fig.add_subplot(235, projection='3d')
ax5.plot_surface(omega_mesh, action_mesh, data, cmap='viridis')
ax5.set_title('TOTAL REWARD (theta=0)')
ax5.set_xlabel('Omega')
ax5.set_ylabel('Action')
ax5.set_zlabel('Reward')

# Plot Total Reward as a 3D Surface plot (omega-action, theta=0)
omega_mesh, action_mesh = np.meshgrid(omega_mesh, action_range)
data = REWARD_SINGLE(theta_fixed_top, omega_mesh, action_mesh)
ax6 = fig.add_subplot(236, projection='3d')
ax6.plot_surface(omega_mesh, action_mesh, data, cmap='viridis')
ax6.set_title('TOTAL REWARD (theta=pi)')
ax6.set_xlabel('Omega')
ax6.set_ylabel('Action')
ax6.set_zlabel('Reward')

plt.show()


# ---------------- M U L T I   T A R G E T   R E W A R D ----------------

# Create a figure
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Multi Target Rewards', fontsize=16)

# Define ranges for all values
theta_range = np.linspace(-np.pi, np.pi, RESOLUTION)
omega_range = np.linspace(-40, 40, RESOLUTION)
action_range = np.array(DISK_ACTIONS)
target_range = NORMALIZE_ANGLE(np.pi + np.linspace(min(MULTI_TARGET_ANGLES), max(MULTI_TARGET_ANGLES), RESOLUTION))

# Fixed values for omega and action
theta_fixed_bottom = 0.0
theta_fixed_top = np.pi
omega_fixed_stopped = 0.0
omega_fixed_full_speed = 0.0
action_fixed_stop = 0.0
action_fixed_full_throttle = DISK_ACTIONS[-1]
target_fixed_middle = 0.0
target_fixed_right = NORMALIZE_ANGLE(np.pi+np.pi/2)

# Plot REWARD_MULTI_THETA
theta_mesh, target_mesh = np.meshgrid(theta_range, target_range)
data = REWARD_MULTI_THETA(theta_mesh, target_mesh)
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot_surface(theta_mesh, target_mesh, data, cmap='viridis')
ax1.set_title('Theta Reward')
ax1.set_xlabel('Theta')
ax1.set_ylabel('Target')
ax1.set_zlabel('Reward')
ax1.grid(True)

# Plot REWARD_MULTI_OMEGA as a 3D surface plot (target=pi+pi/18)
theta_mesh, omega_mesh = np.meshgrid(theta_range, omega_range)
data = REWARD_MULTI_OMEGA(theta_mesh, omega_mesh, target_fixed_right)
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(theta_mesh, omega_mesh, data, cmap='viridis')
ax2.set_title('Omega Reward (target=pi+pi/18)')
ax2.set_xlabel('Theta')
ax2.set_ylabel('Omega')
ax2.set_zlabel('Reward')

# Plot REWARD_MULTI_ACTION
ax3 = fig.add_subplot(233)
data = REWARD_MULTI_ACTION(action_range)
c = data
ax3.plot(action_range, data)
ax3.set_title('Action Reward')
ax3.set_xlabel('Action')
ax3.set_ylabel('Reward')
ax3.grid(True)

# Plot Total Reward as a 3D Surface plot (theta-omega, action=0, target=pi+pi/18)
theta_mesh, omega_mesh = np.meshgrid(theta_range, omega_range)
data = REWARD_MULTI(theta_mesh, omega_mesh, action_fixed_stop, target_fixed_right)
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot_surface(theta_mesh, omega_mesh, data, cmap='viridis')
ax4.set_title('TOTAL REWARD (a=0)')
ax4.set_xlabel('Theta')
ax4.set_ylabel('Omega')
ax4.set_zlabel('Reward')

# Plot Total Reward as a 3D Surface plot (omega-action, theta=0, target=pi+pi/18)
omega_mesh, action_mesh = np.meshgrid(omega_mesh, action_range)
data = REWARD_MULTI(theta_fixed_bottom, omega_mesh, action_mesh, target_fixed_right)
ax5 = fig.add_subplot(235, projection='3d')
ax5.plot_surface(omega_mesh, action_mesh, data, cmap='viridis')
ax5.set_title('TOTAL REWARD (theta=0)')
ax5.set_xlabel('Omega')
ax5.set_ylabel('Action')
ax5.set_zlabel('Reward')

# Plot Total Reward as a 3D Surface plot (omega-action, theta=0, target=pi+pi/18)
omega_mesh, action_mesh = np.meshgrid(omega_mesh, action_range)
data = REWARD_MULTI(theta_fixed_top, omega_mesh, action_mesh, target_fixed_right)
ax6 = fig.add_subplot(236, projection='3d')
ax6.plot_surface(omega_mesh, action_mesh, data, cmap='viridis')
ax6.set_title('TOTAL REWARD (theta=pi)')
ax6.set_xlabel('Omega')
ax6.set_ylabel('Action')
ax6.set_zlabel('Reward')

plt.show()