# system imports
import os
import time
from typing import Dict, Any

# external imports
import torch
import gym
import torch.nn as nn
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.wrappers.time_limit import TimeLimit

# local imports
from rl.agent.base_agent import BaseAgent
from config.definitions import MODELS_DIR
from config.rl import AGENT_REFRESH
from config.env import DISK_ACTIONS

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env, verbose=0, render=False):
        super(CustomCallback, self).__init__(verbose)

        # class attributes
        self.env = env
        self.render = render
        
        # pick timelimit environment wrapper to extract elapsed steps
        self.env_time = self.env
        while not isinstance(self.env_time, TimeLimit):
            self.env_time = self.env_time.env

        # initialize stats
        self.temp_ep_reward = 0
        self.temp_ep_theta_error = 0
        self.temp_max_swing = 0
        self.temp_theta_min = 0
        self.temp_theta_max = 0
        self.temp_ep_max_complete_steps = 0

        # initialize counters
        self.ep = 0
        self.s = 0
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        
        :return: (bool) If the callback returns False, training is aborted early.
        """

        # render the environment if possible
        if self.render:
            self.env.render()

        # extract required variables from locals
        reward = self.locals['rewards'][0]
        info = self.locals['infos'][0]
        done = self.locals['dones'][0]

         # Update stats
        self.temp_ep_reward += reward
        self.temp_theta_max = max(info['theta_bottom'], self.temp_theta_max)
        self.temp_theta_min = min(info['theta_bottom'], self.temp_theta_min)
        self.temp_max_swing = self.temp_theta_max - self.temp_theta_min
        self.temp_ep_theta_error += info['theta_error']
        self.temp_ep_max_complete_steps = max(info['complete_steps'], self.temp_ep_max_complete_steps)

        if done:
            self.ep += 1

            # Log statistics
            self.logger.record('Practical/cum_reward', self.temp_ep_reward)
            self.logger.record('Practical/cum_norm_reward', self.temp_ep_reward/self.env_time._elapsed_steps)
            self.logger.record('Practical/ep_length', self.env_time._elapsed_steps)
            self.logger.record('Practical/cum_theta_error', self.temp_ep_theta_error)
            self.logger.record('Practical/max_complete_steps', self.temp_ep_max_complete_steps)
            self.logger.record('Practical/max_swing', self.temp_max_swing)

            # Reset statistics
            self.temp_ep_reward = 0
            self.temp_theta_min = 0
            self.temp_theta_max = 0
            self.temp_ep_theta_error = 0
            self.temp_ep_max_complete_steps = 0
            self.temp_ep_reward = 0

        self.s += 1
        return True

class A2CBuilt(BaseAgent):
    def __init__(self,
                 policy : str,
                 env,
                 learning_rate : float = 0.0007,
                 n_steps : int = 5,
                 gamma : float = 0.99,
                 ent_coef : float = 0.0,
                 vf_coef : float = 0.5,
                 tensorboard_log : str = None,
                 callbackfeq=100,
                 agent_refresh : float = 1/60,
                 verbose : int = 2,
                 device : str = 'auto',
                 policy_kwargs : Dict[str, Any] | None = None):
        
        self.env = env
        super().__init__(env, 
                         callbackfeq,
                         alpha=learning_rate,
                         gamma=gamma,
                         agent_refresh=agent_refresh)
        
        self.model = sb3.A2C(policy=policy, 
                             env=self.env,
                             learning_rate=learning_rate,
                             n_steps=n_steps,
                             gamma=gamma,
                             ent_coef=ent_coef,
                             vf_coef=vf_coef,
                             tensorboard_log=tensorboard_log,
                             verbose=verbose,
                             device=device,
                             policy_kwargs=policy_kwargs,
                    )
            
    def get_logdir(self):
        return self.model.logger.get_dir()
    
    def learn(self, total_timesteps : int, render : bool = False):
        # set up callback
        self.callback = CustomCallback(env=self.env, 
                                       render=render)

        # start training
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

    def predict(self, obs, deterministic=False):
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.set_parameters(
            load_path_or_dict=os.path.join(MODELS_DIR, path + '.zip')
        )

    def simulate(self, total_timesteps: int):
        # initialize environment
        obs = self.env.reset()

        # initialize stats
        ep = 0
        steps = 0
        ep_cum_reward = 0

        for i in range(total_timesteps):
            # select action
            action = self.predict(obs, deterministic=True)

            # step action
            obs, reward, done, info = self.env.step(action[0])
            self.env.render()
            
            # sleep
            time.sleep(self.agent_refresh)
            
            # update stats
            ep_cum_reward += reward
            steps += 1

            # # log stats
            # self.logger.add_scalar('Output/theta', info['theta'], i)
            # self.logger.add_scalar('Output/omega', info['omega'], i)
            # self.logger.add_scalar('Output/reward', reward, i)
            # self.logger.add_scalar('Input/target_dev', info['target_dev'], i)
            # self.logger.add_scalar('Input/action', DISK_ACTIONS[action], i)

            # terminal state
            print(obs, reward, done)
            if done:
                # log stats
                # self.logger.add_scalar('Validation/cum_reward', ep_cum_reward, ep)
                # self.logger.add_scalar('Validation/ep_length', self.env_time._elapsed_steps, ep)

                # reset stats
                ep_cum_reward / steps
                ep_cum_reward = 0

                ep += 1
                steps = 0

                # reset environment
                self.env.reset()
