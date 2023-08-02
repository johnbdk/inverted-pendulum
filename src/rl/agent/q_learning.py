# system imports
import time
import pickle
import os
from collections import defaultdict

# external imports
import numpy as np


# local imports
from rl.agent.base_agent import BaseAgent

# parameters
PRINT_FREQ = 1000

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
                              epsilon_start=epsilon_start,
                              epsilon_end=epsilon_end,
                              epsilon_decay_steps=epsilon_decay_steps,
                              gamma=gamma,
                              agent_refresh=agent_refresh)
        
        
    def save(self):
        with open(os.path.join(self.log_dir, 'q-table.pkl'), 'wb') as f:
            pickle.dump(dict(self.Qmat), f)

    def load(self, filename):
        with open(os.path.join('models', filename, 'q-table.pkl'), 'rb') as f:
            self.Qmat = pickle.load(f)
    
    def predict(self, obs, exploration=True):
        # exploration
        if exploration and np.random.uniform() < self.epsilon: # exploration (random)
            action = self.env.action_space.sample()

        # exploitation
        else: 
            a = np.array([self.Qmat[obs,i] for i in range(self.env.action_space.n)])
            action = np.random.choice(np.arange(len(a), dtype=int)[a == np.max(a)])
        return action

    def learn(self, total_timesteps : int, callback = None, render : bool = False):
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
            temp_theta_max = max(info['theta_bottom'], temp_theta_max)
            temp_theta_min = min(info['theta_bottom'], temp_theta_min)
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
                self.tb.add_scalar('Parameters/epsilon', self.epsilon, s)
                self.tb.add_scalar('Parameters/alpha', self.alpha, s)
                self.tb.add_scalar('Parameters/gamma', self.gamma, s)
                self.tb.add_scalar('Parameters/nvec', info['nvec'], s)

                self.tb.add_scalar('Practical/ep_length', self.env_time._elapsed_steps, ep)
                self.tb.add_scalar('Practical/max_swing', temp_max_swing, ep)
                self.tb.add_scalar('Practical/cum_reward', temp_ep_reward, ep)
                self.tb.add_scalar('Practical/cum_norm_reward', temp_ep_reward/self.env_time._elapsed_steps, ep)
                self.tb.add_scalar('Practical/cum_theta_error', temp_ep_theta_error, ep)
                self.tb.add_scalar('Practical/max_complete_steps', temp_ep_max_complete_steps, ep)
                
                self.tb.add_scalar('Q-table/new_q_pairs', temp_ep_qpairs, ep)
                self.tb.add_scalar('Q-table/Max_q', temp_ep_max_q, ep)
                self.tb.add_scalar('Q-table/cum_updates', temp_ep_q_change, ep)

                # print info
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
                self.save()

                # reset environment
                obs = self.env.reset()

            else: # not done
                A = reward + self.gamma*max(self.Qmat[obs_new, action_next] for action_next in range(self.env.action_space.n)) - self.Qmat[obs,action]
                self.Qmat[obs,action] += self.alpha*A
                temp_ep_q_change += self.alpha*A
                obs = obs_new

            self.save()

        return self.Qmat
    

    # def simulate(self):
            
    #         self.log_dir = os.path.join('models', 'Jul23_16-23-25_Michalis-Laptop')
    #         Qmat = self.load()
    #         print(Qmat)

    #         obs = self.env.reset()
    #         try:
    #             self.env.render()
    #             done=False
    #             while done==False:
    #                 # pick action according to trained agent
    #                 action = self.__class__.argmax([Qmat[obs, i] for i in range(self.env.action_space.n)])

    #                 # simulation step
    #                 obs, reward, done, info = self.env.step(action)
    #                 print(f'action:{action}, obs:{obs}, done:{done}, reward:{reward}')
    #                 self.env.render()
    #                 # sleep
    #                 time.sleep(self.test_freq)
    #         finally:
    #             self.env.close()