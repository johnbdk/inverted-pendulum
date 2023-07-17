import gym
import numpy as np
from collections import defaultdict
from gym.wrappers.time_limit import TimeLimit

from rl.agent.base_agent import BaseAgent

class QLearning(BaseAgent):
    def __init__(self, 
                 env, 
                 nsteps=5000, 
                 callbackfeq=100, 
                 alpha=0.2, 
                 epsilon=0.9, 
                 gamma=0.99):
        
        super(QLearning, self).__init__(env, 
                              nsteps=nsteps,
                              callbackfeq=callbackfeq, 
                              alpha=alpha, 
                              epsilon=epsilon, 
                              gamma=gamma)

    def run(self):
        Qmat = defaultdict(float) # any new argument set to zero
        env_time = self.env
        while not isinstance(env_time, TimeLimit):
            env_time = env_time.env
        ep_lengths = []
        ep_lengths_steps = []
        
        obs = self.env.reset()
        self.env.render()
        print('goal reached time:')
        for z in range(self.nsteps):

            if z % 100 == 0:
                print('------------ STEP %d/%d ------------' % (z, self.nsteps))

            if np.random.uniform() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.__class__.argmax([Qmat[obs,i] for i in range(self.env.action_space.n)])
            obs_new, reward, done, info = self.env.step(action)
            self.env.render()

            if done and not info.get('TimeLimit.truncated', False): # terminal state = done and not by timeout
                # Saving results:
                print(env_time._elapsed_steps, end=' ')
                ep_lengths.append(env_time._elapsed_steps)
                ep_lengths_steps.append(z)
                
                # Updating Qmat:
                A = reward - Qmat[obs, action] # advantage or TD
                Qmat[obs, action] += self.alpha*A
                obs = self.env.reset()
            else: # Done by timeout or not done
                A = reward + self.gamma*max(Qmat[obs_new, action_next] for action_next in range(self.env.action_space.n)) - Qmat[obs,action]
                Qmat[obs,action] += self.alpha*A
                obs = obs_new
                if info.get('TimeLimit.truncated', False): # done by timeout
                    # Saving results:
                    ep_lengths.append(env_time._elapsed_steps)
                    ep_lengths_steps.append(z)
                    print('out', end=' ')
                    # Reset:
                    obs = self.env.reset()
        print()
        return Qmat, np.array(ep_lengths_steps), np.array(ep_lengths)