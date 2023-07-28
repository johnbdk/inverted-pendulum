from rl.env.custom_env import CustomUnbalancedDisk, Discretizer
from rl.agent.q_learning import QLearning
from rl.agent.dqn import DQN
from gym.wrappers import TimeLimit
import time
import matplotlib.pyplot as plt
import numpy as np

# AGENT_TRAIN_FREQ = 1/24
AGENT_TRAIN_FREQ = 1
AGENT_TEST_FREQ = 1/60

class RLManager():

    def __init__(self, 
                 method='q_learn',
                 nsteps=100_000,
                 max_episode_steps=1_000,
                 nvec=9) -> None:
        
        # class attributes
        self.max_episode_steps = max_episode_steps
        self.nvec = nvec

        # define environment
        self.env = CustomUnbalancedDisk()
        self.env = TimeLimit(self.env, max_episode_steps=max_episode_steps)
        if method == 'q_learn':
            self.env = Discretizer(self.env, nvec=nvec)

        # define agent
        if method == 'q_learn':
            self.agent = QLearning(env=self.env, 
                                   nsteps=nsteps,
                                   callbackfeq=100,
                                   alpha=0.2,
                                   epsilon_start=0.7,
                                   epsilon_end=0.3,
                                   epsilon_decay_steps=0.8*nsteps,
                                   gamma=0.99,
                                   train_freq=AGENT_TRAIN_FREQ,
                                   test_freq=AGENT_TEST_FREQ)
        elif method == 'dqn':
             self.agent = DQN(env=self.env, 
                              nsteps=nsteps,
                              callbackfeq=100,
                              alpha=0.1,
                              epsilon_start=0.9,
                              epsilon_end=0.1,
                              epsilon_decay_steps=0.8*nsteps,
                              gamma=0.99,
                              train_freq=AGENT_TRAIN_FREQ,
                              test_freq=AGENT_TEST_FREQ,
                              buffer_size=5000,
                              batch_size=128,
                              target_update_freq=10000)
        elif method == 'actor_critic':
            self.agent = 0
        else:
            raise ValueError('Unknown method %s' % method)
        
        # reset environment
        self.init_obs = self.env.reset()
    
    def train(self):
        try:
            # start training loop
            self.agent.run()

        finally: #this will always run
            self.env.close()
            
    

    def simulate(self):
        self.agent.simulate()
