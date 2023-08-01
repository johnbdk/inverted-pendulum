from rl.env.custom_unbalanced_disk import CustomUnbalancedDisk, Discretizer
from rl.env.custom_pendulum import CustomPendulum
from rl.agent.q_learning import QLearning
from rl.agent.dqn import DQN
from gym.wrappers import TimeLimit
import time
import matplotlib.pyplot as plt
import numpy as np
import gym
import stable_baselines3 as sb3

# AGENT_TRAIN_FREQ = 1/24
AGENT_TRAIN_FREQ = 1
AGENT_TEST_FREQ = 1/60


from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.env = env

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        
        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.env.render()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class RLManager():

    def __init__(self,
                 method='q_learn',
                 nsteps=500_000,
                 max_episode_steps=1_000,
                 env='unbalanced_disk',
                 nvec=9) -> None:
        
        # class attributes
        self.max_episode_steps = max_episode_steps
        self.nvec = nvec
        self.nsteps = nsteps

        # ----------------- env -----------------

        # define environment
        if env == 'unbalanced_disk':
            self.env = CustomUnbalancedDisk()
        elif env == 'pendulum':
            self.env = CustomPendulum()
        else:
            raise ValueError('Unknown env %s' % env)

        # add time limit
        if max_episode_steps != 0:
            self.env = TimeLimit(self.env, max_episode_steps=max_episode_steps)

        # discretize if necessary
        if method == 'q_learn':
            self.env = Discretizer(self.env, nvec=nvec)

        # ---------------- model ----------------
        # define model
        if method == 'q_learn':
            self.model = QLearning(env=self.env,
                                   callbackfeq=100,
                                   alpha=0.2,
                                   epsilon_start=1.0,
                                   epsilon_end=0,
                                   epsilon_decay_steps=0.5*nsteps,
                                   gamma=0.99,
                                   train_freq=AGENT_TRAIN_FREQ,
                                   test_freq=AGENT_TEST_FREQ)
        elif method == 'dqn':
            self.model = DQN(env=self.env,
                              callbackfeq=100,
                              alpha=0.001,
                              epsilon_start=0.3,
                              epsilon_end=0.0,
                              epsilon_decay_steps=0.5*nsteps,
                              gamma=0.99,
                              train_freq=AGENT_TRAIN_FREQ,
                              test_freq=AGENT_TEST_FREQ,
                              buffer_size=10000,
                              batch_size=64,
                              target_update_freq=10000)
        elif method == 'dqn_built':
            self.model = sb3.DQN(policy='MlpPolicy',
                                 env=self.env,
                                 learning_rate=0.001,
                                 gamma=0.99,
                                 exploration_initial_eps=0.5,
                                 exploration_final_eps=0.0,
                                 exploration_fraction=0.5*nsteps,
                                 buffer_size=50_000,
                                 learning_starts=5000,
                                 batch_size=64,
                                 target_update_interval=10000,
                                 verbose=1,)
        elif method == 'actor_critic':
            self.model = 0
        else:
            raise ValueError('Unknown method %s' % method)
        
        # reset environment
        self.init_obs = self.env.reset()
    
    def train(self):
        try:
            cb = CustomCallback(self.env)
            # start training loop
            self.model.learn(total_timesteps=self.nsteps,
                             callback=cb)

        finally: #this will always run
            self.env.close()
            
    def simulate(self):
        self.model.simulate(total_timesteps=1000)
