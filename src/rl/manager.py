from rl.env.custom_env import CustomUnbalancedDisk, Discretizer
from rl.agent.q_learning import QLearning
from gym.wrappers import TimeLimit
import time
import matplotlib.pyplot as plt

AGENT_TRAIN_FREQ = 1/24
AGENT_TEST_FREQ = 1/60

class RLManager():

    def __init__(self, 
                 method='q_learn',
                 nsteps=1_000_000,
                 max_episode_steps=2_000, 
                 nvec=10) -> None:
        
        # class attributes
        self.max_episode_steps = max_episode_steps
        self.nvec = nvec

        # define environment
        self.env = CustomUnbalancedDisk()
        self.env = TimeLimit(self.env, max_episode_steps=max_episode_steps)
        self.env = Discretizer(self.env, nvec=nvec)

        # define agent
        if method == 'q_learn':
            self.agent = QLearning(env=self.env, 
                                   nsteps=nsteps, 
                                   callbackfeq=100, 
                                   alpha=0.2, 
                                   epsilon=0.2, 
                                   gamma=0.99)
        elif method == 'actor_critic':
            self.agent = 0
        else:
            raise ValueError('Unknown method %s' % method)
        
        # reset environment
        self.init_obs = self.env.reset()
    
    def train(self):
        try:
            Qmat, ep_lengths_steps, ep_lengths = self.agent.run()
            print(ep_lengths_steps, ep_lengths)
            plt.plot(ep_lengths_steps, self.agent.__class__.roll_mean(ep_lengths, start=self.max_episode_steps),label=str(self.nvec)) #c)
            plt.show()
        finally: #this will always run
            self.env.close()
            
    def simulate(self):
        obs = self.env.reset()
        try:
            self.env.render()
            done=False
            while done==False:
                # pick action according to trained agent
                # action = argmax([Qmat[obs, i] for i in range(self.env.action_space.n)])
                action = self.env.action_space.sample()
                # action = 3

                # simulation step
                obs, reward, done, info = self.env.step(action) # TODO change this
                self.env.render()

                # sleep
                time.sleep(AGENT_TEST_FREQ)
                
                # print(obs, reward, action, done, info) #check info on timelimit
        finally:
            self.env.close()


