# imports
import numpy as np


class BaseAgent(object):
    def __init__(self, env, nsteps=5000, callbackfeq=100, alpha=0.2, epsilon=0.2, gamma=0.99):
        self.env = env
        self.nsteps = nsteps
        self.callbackfeq = callbackfeq
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def run(self):
        pass

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
