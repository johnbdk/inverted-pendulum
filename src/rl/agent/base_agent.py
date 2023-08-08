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
from config.env import DISK_ACTIONS

class BaseAgent(object):
    def __init__(self, 
                 env,
                 callbackfeq=100,
                 alpha=0.2,
                 gamma=0.99,
                 agent_refresh=1/60):
        
        self.env = env

        # pick timelimit environment wrapper to extract elapsed steps
        self.env_time = self.env
        while not isinstance(self.env_time, TimeLimit):
            self.env_time = self.env_time.env

        self.callbackfeq = callbackfeq
        self.alpha = alpha
        
        self.gamma = gamma
        self.agent_refresh = agent_refresh

    
    def setup_logger(self):
        # start tensorboard session
        self.log_dir = os.path.join(MODELS_DIR, self.__class__.__name__ + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.logger = SummaryWriter(log_dir=self.log_dir)

    def get_logdir(self):
        return self.logger.log_dir

    # abstract method for training
    def learn(self, total_timesteps : int, render : bool = False):
        pass

    # abstract method for predicting action
    def predict(self, obs, deterministic=False):
        pass

    # abstract method for saving model
    def save(self, path : str):
        pass

    # abstract method for loading model
    def load(self):
        pass

    # method for testing (override in extended classes if necessary)
    def simulate(self, total_timesteps : int):
        # initialize environment
        obs = self.env.reset()

        # initialize stats
        ep = 0
        steps = 0
        ep_cum_reward = 0
        
        for i in range(total_timesteps):
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
            self.logger.add_scalar('Output/theta', info['theta'], i)
            self.logger.add_scalar('Output/omega', info['omega'], i)
            self.logger.add_scalar('Output/reward', reward, i)
            self.logger.add_scalar('Input/target_dev', info['target_dev'], i)
            self.logger.add_scalar('Input/action', action, i)

            # terminal state
            if done:
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
