# system imports
import os
from datetime import datetime

# external imports
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers.time_limit import TimeLimit

class BaseAgent(object):
    def __init__(self, 
                 env,
                 callbackfeq=100,
                 alpha=0.2,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=0.9*5000,
                 gamma=0.99,
                 train_freq=1/24,
                 test_freq=1/60):
        
        self.env = env

        # pick timelimit environment wrapper to extract elapsed steps
        self.env_time = self.env
        while not isinstance(self.env_time, TimeLimit):
            self.env_time = self.env_time.env

        self.callbackfeq = callbackfeq
        self.alpha = alpha
        self.epsilon_start=epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_end=epsilon_end
        self.epsilon_decay_steps=epsilon_decay_steps
        self.gamma = gamma
        self.train_freq = train_freq
        self.test_freq = test_freq

        # start tensorboard session
        self.log_dir = os.path.join('runs', self.__class__.__name__ + '_' + datetime.now().isoformat())
        self.tb = SummaryWriter(log_dir=self.log_dir)

        
    # abstract method for training
    def learn(self, total_timesteps : int, callback = None, render : bool = False):
        pass

    # abstract method for predicting action
    def predict(self, obs, exploration=True):
        pass

    # abstract method for saving model
    def save(self):
        pass

    # abstract method for loading model
    def load(self):
        pass

    # method for testing (override in extended classes if necessary)
    def simulate(self, total_timesteps : int):
        obs = self.env.reset()
        try:
            for _ in range(total_timesteps):
                action, _ = self.predict(obs)
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                self.time.sleep(self.test_freq)
                if done:
                    self.env.reset()
        finally:
            self.env.close()


    
