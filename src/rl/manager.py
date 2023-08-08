# system imports
import os
import time

# external imports
from gym.wrappers import TimeLimit
from torch import nn
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

# local imports
from rl.env.unbalanced_disk.discretizer import Discretizer
from rl.env.unbalanced_disk.custom_unbalanced_disk import CustomUnbalancedDiskSingle, CustomUnbalancedDiskMulti
from rl.env.pendulum.custom_pendulum import CustomPendulum
from rl.agent.q_learning import QLearning
from rl.agent.dqn import DQN
from rl.agent.a2c import A2C
from rl.agent.a2c_built import A2CBuilt
from config.env import NVEC
from config.definitions import MODELS_DIR
from config.rl import (
    TRAIN_STEPS,
    TEST_STEPS,
    TEST_CALLBACK_FREQ,
    SAVE_FREQ,
    AGENT_REFRESH,
    EPSILON_PARAMS,
    QLEARN_PARAMS,
    DQN_PARAMS,
    ACTOR_CRITIC_PARAMS,
    MAX_EPISODE_STEPS,
    STATE_SPACE_MAP,
    ACTION_SPACE_MAP
)


class RLManager():
    def __init__(self,
                 env : str = 'unbalanced_disk',
                 method : str ='q_learn',
                 mode : str ='train',
                 multi_target : bool = False,
                 model_path : str | None = None,
                 ) -> None:
        
        # class attributes
        self.mode = mode
        self.method = method
        self.total_timesteps = TRAIN_STEPS if mode == 'train' else TEST_STEPS

        # ----------------- env -----------------

        # define environment
        if env == 'unbalanced_disk':
            if not multi_target or method in ['q_learn', 'dqn']: 
                print('Task: Single target')
                self.env = CustomUnbalancedDiskSingle(action_space_type=ACTION_SPACE_MAP[method])
            elif multi_target: # only in A2C
                print('Task: Multi target')
                self.env = CustomUnbalancedDiskMulti()
        elif env == 'pendulum':
            self.env = CustomPendulum()
        else:
            raise ValueError('Invalid env configuration %s' % env)

        # add time limit
        self.env = TimeLimit(self.env, max_episode_steps=MAX_EPISODE_STEPS)

        # discretize if necessary
        if not multi_target and STATE_SPACE_MAP[method] == 'discrete':
            print('Starting discretization')
            self.env = Discretizer(self.env, nvec=NVEC)
        
        # ---------------- model ----------------
        # define model
        if method == 'q_learn':
            print('Loading Q-Learning')
            self.model = QLearning(env=self.env,
                                   callbackfeq=TEST_CALLBACK_FREQ,
                                   alpha=QLEARN_PARAMS['alpha'],
                                   epsilon_start=EPSILON_PARAMS['epsilon_start'],
                                   epsilon_end=EPSILON_PARAMS['epsilon_end'],
                                   epsilon_decay_steps=EPSILON_PARAMS['epsilon_decay_steps'],
                                   gamma=QLEARN_PARAMS['gamma'],
                                   agent_refresh=AGENT_REFRESH)
        elif method == 'dqn':
            print('Loading DQN')
            self.model = DQN(env=self.env,
                              callbackfeq=TEST_CALLBACK_FREQ,
                              alpha=DQN_PARAMS['learning_rate'],
                              epsilon_start=EPSILON_PARAMS['epsilon_start'],
                              epsilon_end=EPSILON_PARAMS['epsilon_end'],
                              epsilon_decay_steps=EPSILON_PARAMS['epsilon_decay_steps'],
                              gamma=DQN_PARAMS['gamma'],
                              agent_refresh=AGENT_REFRESH,
                              hidden_layers=DQN_PARAMS['hidden_layers'],
                              buffer_size=DQN_PARAMS['buffer_size'],
                              batch_size=DQN_PARAMS['batch_size'],
                              target_update_freq=DQN_PARAMS['target_update_freq'],)
        elif method == 'a2c':
            print('Loading A2C')
            self.model = A2C(env=self.env,
                             callbackfeq=TEST_CALLBACK_FREQ,
                             gamma=ACTOR_CRITIC_PARAMS['gamma'],
                             learning_rate=ACTOR_CRITIC_PARAMS['learning_rate'],
                             alpha_entropy=ACTOR_CRITIC_PARAMS['alpha_entropy'],
                             alpha_actor=ACTOR_CRITIC_PARAMS['alpha_actor'],
                             rollout_length=ACTOR_CRITIC_PARAMS['rollout_length'],
                             agent_refresh=AGENT_REFRESH)
        elif method == 'a2c_built':
            print('Loading A2C built')
            self.model = A2CBuilt(policy="MlpPolicy", 
                                  env=self.env,
                                  learning_rate=ACTOR_CRITIC_PARAMS['learning_rate'],
                                  n_steps=ACTOR_CRITIC_PARAMS['rollout_length'],
                                  gamma=ACTOR_CRITIC_PARAMS['gamma'],
                                  ent_coef=ACTOR_CRITIC_PARAMS['alpha_entropy'],
                                  vf_coef=ACTOR_CRITIC_PARAMS['alpha_actor'],
                                  tensorboard_log=MODELS_DIR,
                                  verbose=1,
                                  device='cpu', # much faster than GPU
                                #   policy_kwargs= dict(activation_fn=nn.ReLU,
                                #                       net_arch=dict(
                                #                         pi=[32, 32], 
                                #                         vf=[32, 32]))
            )
        else:
            raise ValueError('Unknown method %s' % method)
        
        # load model
        if mode == 'test':
            assert model_path is not None
            print('Loading model %s' % model_path)
            self.model.load(path=model_path)

        # reset environment
        # self.init_obs = self.env.reset()

    def train(self, render=False):
        print('Starting train')
        try:
            # start training loop
            self.model.learn(total_timesteps=self.total_timesteps, render=render)
        finally: # Always run this
            # save model
            self.model.save(path=self.model.get_logdir())
            # close environment
            self.env.close()
            
    def simulate(self):
        print('Starting simulation')
        try:
            self.model.simulate(total_timesteps=self.total_timesteps)
        except KeyboardInterrupt:
            print("Keyboard interruption.")
        finally:
            self.env.close()
       