from rl.env.custom_env import CustomUnbalancedDisk
import time

AGENT_TRAIN_FREQ = 1/24
AGENT_TEST_FREQ = 1/60

class RLManager():

    def __init__(self, method='q_learn') -> None:
        # define environment
        self.env = CustomUnbalancedDisk()

        # define agent
        if method == 'q_learn':
            self.agent = 0
        elif method == 'actor_critic':
            self.agent = 1
        else:
            raise ValueError('Unknown method %s' % method)
        
        # reset environment
        self.init_obs = self.env.reset()
    
    def train(self, steps=200):
        try:
            for i in range(steps):
                obs, reward, done, info = self.env.step(self.env.action_space.sample()) #random action
                print(obs,reward)
                self.env.render()
                time.sleep(AGENT_TRAIN_FREQ)
                if done:
                    obs = self.env.reset()
        finally: #this will always run
            self.env.close()
            
    def simulate(self):
        obs = self.env.reset()
        try:
            self.env.render()
            done=False
            while done==False:
                # action = argmax([Qmat[obs, i] for i in range(self.env.action_space.n)])
                action = self.env.step(self.env.action_space.sample()) # TODO change this
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                time.sleep(AGENT_TEST_FREQ)
                print(obs, reward, action, done, info) #check info on timelimit
                #check on info['TimeLimit.truncated']
        finally:
            self.env.close()