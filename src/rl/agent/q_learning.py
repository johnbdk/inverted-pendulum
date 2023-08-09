# system imports
import time
import pickle
import os
from collections import defaultdict

# external imports
import numpy as np

# local imports
from rl.agent.base_agent import BaseAgent
from config.definitions import MODELS_DIR


class QLearning(BaseAgent):
    def __init__(self, 
                 env,
                 callbackfeq=100, 
                 alpha=0.2, 
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=0.9*5000,
                 gamma=0.99,
                 agent_refresh=1/60):
        
        super(QLearning, self).__init__(env,
                              callbackfeq=callbackfeq, 
                              alpha=alpha,
                              gamma=gamma,
                              agent_refresh=agent_refresh)
        """
        Constructor for the QLearning class.
        
        :param env: The environment to interact with.
        :param callbackfeq: Frequency of callbacks. Defaults to 100.
        :param alpha: Learning rate. Defaults to 0.2.
        :param epsilon_start: Initial epsilon for the epsilon-greedy policy. Defaults to 1.0.
        :param epsilon_end: Final epsilon for the epsilon-greedy policy. Defaults to 0.1.
        :param epsilon_decay_steps: Number of steps for epsilon decay. Defaults to 0.9 * 5000.
        :param gamma: Discount factor. Defaults to 0.99.
        :param agent_refresh: Refresh rate of the agent. Defaults to 1/60.
        
        Initializes the Q-learning agent with the provided parameters.
        """
        
        # extra class attributes
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_end=epsilon_end
        self.epsilon_decay_steps=epsilon_decay_steps

        # setup logger
        self.setup_logger()

    def learn(self, total_timesteps : int, render : bool = False):
        """
        Method to train the Q-learning agent.
        
        :param total_timesteps: Total number of timesteps for training.
        :param render: Flag to enable rendering. Defaults to False.
        
        Initializes the Q-table and trains the agent using the Q-learning algorithm.
        """

        # initialize Q-table
        init_q_value = 0
        self.Qmat = defaultdict(lambda: init_q_value) # any new argument set to zero
        
        # initialize stats (long term)
        step_max_q = []
        ep = 0
        action = 0

        # initialize temporary stats (short term)
        temp_ep_reward = 0
        temp_ep_qpairs = 0
        temp_ep_max_q = 0
        temp_ep_theta_error = 0
        temp_max_swing = 0
        temp_theta_min = 0
        temp_theta_max = 0
        temp_ep_q_change = 0
        temp_ep_max_complete_steps = 0

        # initialize environment
        obs = self.env.reset()
        if render:
            self.env.render()

        # train loop
        for s in range(total_timesteps):

            self.epsilon = max(self.epsilon_start - s * ((self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps), self.epsilon_end)

            # select action using exploration-exploitation policy
            action = self.predict(obs)

            # step action in environment
            obs_new, reward, done, info = self.env.step(action)
            if render:
                self.env.render()

            # append stats
            temp_ep_reward += reward
            if self.Qmat.values():
                temp_max_q = max(self.Qmat.values())
                step_max_q.append(temp_max_q)
            if self.Qmat[obs, action] == init_q_value:
                temp_ep_qpairs += 1
            temp_theta_max = max(info['theta'], temp_theta_max)
            temp_theta_min = min(info['theta'], temp_theta_min)
            temp_max_swing = temp_theta_max - temp_theta_min
            temp_ep_theta_error += info['theta_error']
            temp_ep_max_complete_steps = max(info['complete_steps'], temp_ep_max_complete_steps)

            # check for terminating condition
            if done:
                print('\n---- Episode %d Completed ----' % (ep))
                if not info.get('TimeLimit.truncated', False): # done by reaching target
                    print('reason: success')
                    # update self.Qmat
                    A = reward - self.Qmat[obs, action] # advantage or TD
                    self.Qmat[obs, action] += self.alpha*A
                    temp_ep_q_change += self.alpha*A
                else: # done by timeout
                    print('reason: timeout')
                    # update self.Qmat
                    A = reward + self.gamma*max(self.Qmat[obs_new, action_next] for action_next in range(self.env.action_space.n)) - self.Qmat[obs,action]
                    self.Qmat[obs,action] += self.alpha*A
                    temp_ep_q_change += self.alpha*A
                    obs = obs_new
                    
                ep += 1
   
                # save stats
                temp_ep_max_q = np.max(step_max_q[-self.env_time._elapsed_steps:])
                self.logger.add_scalar('Parameters/epsilon', self.epsilon, s)
                self.logger.add_scalar('Parameters/alpha', self.alpha, s)
                self.logger.add_scalar('Parameters/gamma', self.gamma, s)
                self.logger.add_scalar('Parameters/nvec', info['nvec'], s)

                self.logger.add_scalar('Practical/ep_length', self.env_time._elapsed_steps, s)
                self.logger.add_scalar('Practical/max_swing', temp_max_swing, s)
                self.logger.add_scalar('Practical/cum_reward', temp_ep_reward, s)
                self.logger.add_scalar('Practical/cum_norm_reward', temp_ep_reward/self.env_time._elapsed_steps, s)
                self.logger.add_scalar('Practical/cum_theta_error', temp_ep_theta_error, s)
                self.logger.add_scalar('Practical/max_complete_steps', temp_ep_max_complete_steps, s)
                
                self.logger.add_scalar('Q-table/new_q_pairs', temp_ep_qpairs, s)
                self.logger.add_scalar('Q-table/Max_q', temp_ep_max_q, s)
                self.logger.add_scalar('Q-table/cum_updates', temp_ep_q_change, s)

                # print info
                print('steps: %d/%d (%.2f%%)' % (s, total_timesteps, (100*s)/total_timesteps))
                print('reward: %.2f' % (temp_ep_reward))
                print('length: %d' % (self.env_time._elapsed_steps))
                print('max_q: %.2f' % (temp_ep_max_q))
                print('new_q_pairs: %d' % (temp_ep_qpairs))
                print('max_swing: %.2f' % (temp_max_swing))
                print('accumulated angle error: %.2f' % (temp_ep_theta_error))
                print('accumulated q-table updates: %.2f' % (temp_ep_q_change))
                print('max complete steps: %d' % (temp_ep_max_complete_steps))
                print('---------------------------')

                # reset stats
                temp_ep_reward = 0
                temp_ep_qpairs = 0
                temp_theta_min = 0
                temp_theta_max = 0
                temp_ep_theta_error = 0
                temp_ep_q_change = 0
                temp_ep_max_complete_steps = 0

                # save q-table
                self.save(path=self.get_logdir())

                # reset environment
                obs = self.env.reset()

            else: # not done
                A = reward + self.gamma*max(self.Qmat[obs_new, action_next] for action_next in range(self.env.action_space.n)) - self.Qmat[obs,action]
                self.Qmat[obs,action] += self.alpha*A
                temp_ep_q_change += self.alpha*A
                obs = obs_new

            self.save(path=self.get_logdir())
            
        return self.Qmat
    
    def predict(self, obs, deterministic=False):
        """
        Method to predict an action based on an observation.
        
        :param obs: Observation from the environment.
        :param deterministic: Flag to determine if the prediction is deterministic. Defaults to False.
        :return: Predicted action.
        
        Uses the epsilon-greedy policy for exploration and exploitation.
        """

        # exploration
        if not deterministic and np.random.uniform() < self.epsilon: # exploration (random)
            action = self.env.action_space.sample()

        # exploitation
        else: 
            a = np.array([self.Qmat[obs,i] for i in range(self.env.action_space.n)])
            action = np.random.choice(np.arange(len(a), dtype=int)[a == np.max(a)])
        return action
    
    def save(self, path):
        """
        Method to save the Q-table.
        
        :param path: Path to save the Q-table.
        
        Saves the Q-table to the specified path.
        """

        with open(os.path.join(path, 'q-table.pkl'), 'wb') as f:
            pickle.dump(dict(self.Qmat), f)

    def load(self, path):
        """
        Method to load the Q-table.
        
        :param path: Path to load the Q-table from.
        
        Loads the Q-table from the specified path.
        """
        with open(os.path.join(MODELS_DIR, path, 'q-table.pkl'), 'rb') as f:
            self.Qmat = pickle.load(f)
