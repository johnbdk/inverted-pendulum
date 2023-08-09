# system imports
import os
from datetime import datetime
import time

# external imports
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers.time_limit import TimeLimit

# local imports
from config.definitions import MODELS_DIR
from config.rl import TEST_EPISODES

class BaseAgent(object):
    def __init__(self, 
                 env,
                 callbackfeq=100,
                 alpha=0.2,
                 gamma=0.99,
                 agent_refresh=1/60):
        """
        Constructor for the BaseAgent class.
        
        :param env: The environment to interact with.
        :param callbackfeq: Frequency of callbacks. Defaults to 100.
        :param alpha: Learning rate. Defaults to 0.2.
        :param gamma: Discount factor. Defaults to 0.99.
        :param agent_refresh: Refresh rate of the agent. Defaults to 1/60.
        
        Initializes the agent with the provided parameters.
        """
        
        # class attributes
        self.env = env
        self.callbackfeq = callbackfeq
        self.alpha = alpha
        self.gamma = gamma
        self.agent_refresh = agent_refresh

        # pick timelimit environment wrapper to extract elapsed steps
        self.env_time = self.env
        while not isinstance(self.env_time, TimeLimit):
            self.env_time = self.env_time.env

    
    def setup_logger(self):
        """
        Method to set up the logger for tensorboard session.
        """

        # generate the log path
        self.log_dir = os.path.join(MODELS_DIR, self.__class__.__name__ + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        # start tensorboard session
        self.logger = SummaryWriter(log_dir=self.log_dir)

    def get_logdir(self):
        """
        Method to get the log directory.
        
        :return: Path to the log directory.
        """
        return self.logger.log_dir

    def learn(self, total_timesteps : int, render : bool = False):
        """
        Abstract method for training the agent.
        
        :param total_timesteps: Total number of timesteps for training.
        :param render: Flag to enable rendering. Defaults to False.
        """
        pass

    def predict(self, obs, deterministic=False):
        """
        Abstract method for predicting an action based on an observation.
        
        :param obs: Observation from the environment.
        :param deterministic: Flag to determine if the prediction is deterministic. Defaults to False.
        :return: Predicted action.
        """
        pass

    def save(self, path : str):
        """
        Abstract method for saving the model.
        
        :param path: Path to save the model.
        """
        pass

    def load(self):
        """
        Abstract method for loading the model.
        """
        pass

    def simulate(self):
        """
        Method for testing the agent (can be overridden in extended classes if necessary).
        
        Simulates the agent's interactions with the environment and logs relevant statistics.
        """
        # initialize environment
        obs = self.env.reset()

        # initialize stats
        ep = 0
        steps = 0
        ep_cum_reward = 0
        done = False
        
        while not done:
            # select action
            action = self.predict(obs, deterministic=True)

            # step action
            obs, reward, done, info = self.env.step(action)
            self.env.render()

            # sleep
            time.sleep(self.agent_refresh)
            
            # update stats
            ep_cum_reward += reward
            steps += 1

            # log stats
            self.logger.add_scalar('Output/theta', info['theta'], steps)
            self.logger.add_scalar('Output/omega', info['omega'], steps)
            self.logger.add_scalar('Output/reward', reward, steps)
            self.logger.add_scalar('Input/target_dev', info['target_dev'], steps)
            self.logger.add_scalar('Input/action', action, steps)

        # log stats
        self.logger.add_scalar('Validation/cum_reward', ep_cum_reward, ep)
        self.logger.add_scalar('Validation/ep_length', self.env_time._elapsed_steps, ep)

        # reset stats
        ep_cum_reward / steps
        ep_cum_reward = 0

        ep += 1
        steps = 0

        # reset environment
        self.env.reset()
