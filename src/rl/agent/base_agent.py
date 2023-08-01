# imports
import numpy as np


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
        self.callbackfeq = callbackfeq
        self.alpha = alpha
        self.epsilon_start=epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_end=epsilon_end
        self.epsilon_decay_steps=epsilon_decay_steps
        self.gamma = gamma
        self.train_freq = train_freq
        self.test_freq = test_freq

    # abstract method for training
    def learn(self, total_timesteps : int, callback = None):
        pass

    # abstract method for predicting action
    def predict(self, observation):
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

    @staticmethod
    def argmax(a):
        # Random argmax
        a = np.array(a)
        return np.random.choice(np.arange(len(a), dtype=int)[a == np.max(a)])
    
    @staticmethod
    def roll_mean(ar, start=2000, N=50): # smoothing if needed
        s = 1 - 1/N
        k = start
        out = np.zeros(ar.shape)
        for i, a in enumerate(ar):
            k = s*k + (1-s)*a
            out[i] = k
        return out
