# system imports
import time
import pickle
import os

# external imports

import numpy as np
import gym
from collections import defaultdict
from gym.wrappers.time_limit import TimeLimit
from torch.utils.tensorboard import SummaryWriter

# local imports
from rl.agent.base_agent import BaseAgent

# parameters
PRINT_FREQ = 1000

class QLearning(BaseAgent):
    def __init__(self, 
                 env, 
                 nsteps=5000, 
                 callbackfeq=100, 
                 alpha=0.2, 
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=0.9*5000,
                 gamma=0.99,
                 train_freq=1/24,
                 test_freq=1/60):
        
        super(QLearning, self).__init__(env, 
                              nsteps=nsteps,
                              callbackfeq=callbackfeq, 
                              alpha=alpha, 
                              epsilon_start=epsilon_start,
                              epsilon_end=epsilon_end,
                              epsilon_decay_steps=epsilon_decay_steps,
                              gamma=gamma,
                              train_freq=train_freq,
                              test_freq=test_freq)
        
        # start tensorboard session
        self.tb = SummaryWriter()
        self.log_dir = self.tb.log_dir

    def save(self, Qmat):
        with open(os.path.join(self.log_dir, 'q-table.pkl'), 'wb') as f:
            pickle.dump(Qmat, f)

    def run(self):
        # initialize Q-table
        init_q_value = 0
        Qmat = defaultdict(lambda: init_q_value) # any new argument set to zero
        
        # pick timelimit environment wrapper to extract elapsed steps
        env_time = self.env
        while not isinstance(env_time, TimeLimit):
            env_time = env_time.env

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
        temp_ep_actions_taken = []
        transition_counts = np.zeros((self.env.action_space.n, self.env.action_space.n))

        # initialize environment
        obs = self.env.reset()
        self.env.render()

        # train loop
        for s in range(self.nsteps):

            self.epsilon = max(self.epsilon_start - s * ((self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps), self.epsilon_end)

            # select action using exploration-exploitation policy
            prev_action = action
            if np.random.uniform() < self.epsilon: # exploration (random)
                action = self.env.action_space.sample()
            else: # exploitation (argmax)
                action = self.__class__.argmax([Qmat[obs,i] for i in range(self.env.action_space.n)])
            action = [-3, -1, 0, 1, 3][action]

            # step action in environment
            obs_new, reward, done, info = self.env.step(action)
            self.env.render()

            # append stats
            temp_ep_reward += reward
            if Qmat.values():
                temp_max_q = max(Qmat.values())
                step_max_q.append(temp_max_q)
            if Qmat[obs, action] == init_q_value:
                temp_ep_qpairs += 1
            temp_theta_max = max(info['observation'][0], temp_theta_max)
            temp_theta_min = min(info['observation'][0], temp_theta_min)
            temp_max_swing = temp_theta_max - temp_theta_min
            temp_ep_theta_error += np.pi - np.abs(info['observation'][0])
            temp_ep_max_complete_steps = max(info['complete_steps'], temp_ep_max_complete_steps)
            temp_ep_actions_taken.append(action)
            transition_counts[prev_action, action] += 1

            
            # print('theta', info['observation'][0])
            # print(temp_theta_min, temp_theta_max, temp_max_swing)

            # check for terminating condition
            if done:
                print('\n---- Episode %d Completed ----' % (ep))
                if not info.get('TimeLimit.truncated', False): # done by reaching target
                    print('reason: success')
                    # update Qmat
                    A = reward - Qmat[obs, action] # advantage or TD
                    Qmat[obs, action] += self.alpha*A
                    temp_ep_q_change += self.alpha*A
                else: # done by timeout
                    print('reason: timeout')
                    # update Qmat
                    A = reward + self.gamma*max(Qmat[obs_new, action_next] for action_next in range(self.env.action_space.n)) - Qmat[obs,action]
                    Qmat[obs,action] += self.alpha*A
                    temp_ep_q_change += self.alpha*A
                    obs = obs_new
                    
                ep += 1
   
                # save stats
                temp_ep_max_q = np.max(step_max_q[-env_time._elapsed_steps:])
                transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
                transition_probs = (transition_probs - np.min(transition_probs)) / (np.max(transition_probs) - np.min(transition_probs))
                transition_image = np.repeat(np.expand_dims(transition_probs, axis=0), 3, axis=0)
                self.tb.add_scalar('Parameters/epsilon', self.epsilon, ep)
                self.tb.add_scalar('Parameters/alpha', self.alpha, ep)
                self.tb.add_scalar('Parameters/gamma', self.gamma, ep)
                self.tb.add_scalar('Parameters/nvec', info['nvec'], ep)
                self.tb.add_scalar('Practical/ep_length', env_time._elapsed_steps, ep)
                self.tb.add_histogram('Practical/actions_distribution', np.array(temp_ep_actions_taken), ep)
                self.tb.add_image('Practical/action_transitions', transition_image, ep)
                self.tb.add_scalar('Practical/max_swing', temp_max_swing, ep)
                self.tb.add_scalar('Practical/cum_reward', temp_ep_reward, ep)
                self.tb.add_scalar('Practical/cum_theta_error', temp_ep_theta_error, ep)
                self.tb.add_scalar('Practical/max_complete_steps', temp_ep_max_complete_steps, ep)
                self.tb.add_scalar('Q-table/new_q_pairs', temp_ep_qpairs, ep)
                self.tb.add_scalar('Q-table/Max_q', temp_ep_max_q, ep)
                self.tb.add_scalar('Q-table/cum_updates', temp_ep_q_change, ep)

                # print info
                print('reward: %.2f' % (temp_ep_reward))
                print('length: %d' % (env_time._elapsed_steps))
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
                temp_ep_actions_taken = []
                transition_counts = np.zeros((self.env.action_space.n, self.env.action_space.n))

                # save q-table
                self.save(dict(Qmat))

                # reset environment
                obs = self.env.reset()

            else: # not done
                A = reward + self.gamma*max(Qmat[obs_new, action_next] for action_next in range(self.env.action_space.n)) - Qmat[obs,action]
                Qmat[obs,action] += self.alpha*A
                temp_ep_q_change += self.alpha*A
                obs = obs_new

            self.save(dict(Qmat))

            # Agent sleep if necessary
            # time.sleep(self.train_freq)

            # print stats
            # if z % PRINT_FREQ == 0:
            #     theta_deviation = info['theta_deviation']
            #     omega_deviation = info['omega_deviation']
            #     throttle_deviation = info['throttle_deviation']
            #     print('[%.0fK/%.0fK] action:%d, reward:%.2f, R_theta:%.2f, R_omega:%.2f, R_throttle:%.2f' % (z/1000, self.nsteps/1000, action, reward, -theta_deviation, -omega_deviation, -throttle_deviation))

        return Qmat