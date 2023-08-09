# system imports
import os
import time
from typing import Dict, Any

# external imports
import torch.nn as nn
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback

# local imports
from rl.agent.base_agent import BaseAgent
from config.definitions import MODELS_DIR
from config.rl import TEST_EPISODES

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env, verbose=0, render=False):
        """
        A custom callback for logging statistics during training.
        
        :param env: The environment to interact with.
        :param verbose: Verbosity level, 0 for no output, 1 for info messages, 2 for debug messages.
        :param render: Flag to enable rendering.

        Initializes the custom callback with statistics and counters.
        """

        super(CustomCallback, self).__init__(verbose)

        # class attributes
        self.env = env
        self.render = render

        # initialize stats
        self.temp_ep_reward = 0
        self.temp_ep_theta_error = 0
        self.temp_max_swing = 0
        self.temp_theta_min = 0
        self.temp_theta_max = 0
        self.temp_ep_max_complete_steps = 0

        self.elapsed_steps = 0

        # initialize counters
        self.ep = 0
        self.s = 0
        
    def _on_step(self) -> bool:
        """
        Method called by the model after each call to `env.step()`.
        
        :return: (bool) If the callback returns False, training is aborted early.

        Performs logging and statistics updates for each step.
        """

        # render the environment if possible
        if self.render:
            self.env.render()

        # extract required variables from locals
        reward = self.locals['rewards'][0]
        info = self.locals['infos'][0]
        done = self.locals['dones'][0]

        # update stats
        self.temp_ep_reward += reward
        self.temp_theta_max = max(info['theta'], self.temp_theta_max)
        self.temp_theta_min = min(info['theta'], self.temp_theta_min)
        self.temp_max_swing = self.temp_theta_max - self.temp_theta_min
        self.temp_ep_theta_error += info['theta_error']
        self.temp_ep_max_complete_steps = max(info['complete_steps'], self.temp_ep_max_complete_steps)

        self.elapsed_steps += 1

        if done:
            self.ep += 1

            # Log statistics
            self.logger.record('Practical/cum_reward', self.temp_ep_reward)
            self.logger.record('Practical/cum_norm_reward', self.temp_ep_reward/self.elapsed_steps)
            self.logger.record('Practical/ep_length', self.elapsed_steps)
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

            self.elapsed_steps = 0

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
        """
        Constructor for the A2CBuilt class, representing a built-in A2C agent.
        
        :param policy: Policy architecture to use (e.g. 'MlpPolicy').
        :param env: The environment to interact with.
        :param learning_rate, n_steps, gamma, ent_coef, vf_coef: A2C hyperparameters.
        :param tensorboard_log: Directory for TensorBoard logs.
        :param callbackfeq, agent_refresh: Parameters inherited from BaseAgent.
        :param verbose: Verbosity level.
        :param device: Device to run on ('auto', 'cuda', 'cpu').
        :param policy_kwargs: Additional keyword arguments for the policy.

        Initializes the A2C agent with the provided parameters.
        """
        
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
        """
        :return: Logging directory path.
        """

        return self.model.logger.get_dir()
    
    def learn(self, total_timesteps : int, render : bool = False):
        """
        Method to train the A2C agent.
        
        :param total_timesteps: Total number of timesteps for training.
        :param render: Flag to enable rendering. Defaults to False.
        
        Trains the agent using the A2C algorithm.
        """

        # set up callback
        self.callback = CustomCallback(env=self.env, 
                                       render=render)

        # start training
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

    def predict(self, obs, deterministic=False):
        """
        Method to predict an action based on an observation.
        
        :param obs: Observation from the environment.
        :param deterministic: Flag to determine if the prediction is deterministic. Defaults to False.
        :return: Predicted action.
        
        Uses the A2C model to predict the action.
        """
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path):
        """
        Method to save the A2C model.
        
        :param path: Path to save the model.
        
        Saves the model to the specified path.
        """

        self.model.save(path)

    def load(self, path):
        """
        Method to load the A2C model.
        
        :param path: Path to load the model from.
        
        Loads the model from the specified path.
        """

        self.model.set_parameters(
            load_path_or_dict=os.path.join(MODELS_DIR, path + '.zip')
        )

    def simulate(self):
        """
        Method to simulate the A2C agent.

        Simulates the agent's behavior, logs statistics, and renders the environment.
        """
        
        # initialize environment
        obs = self.env.reset()

        # setup logger
        self.setup_logger()

        # initialize stats
        steps = 0
        ep_cum_reward = 0
        done = False

        for ep in range(TEST_EPISODES):

            while not done:
                # select action
                action = self.predict(obs, deterministic=True)[0]

                # step action
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                
                # sleep
                time.sleep(self.agent_refresh)

                # update stats
                ep_cum_reward += reward
                steps += 1

                # log stats
                self.logger.add_scalar('Output/theta', info['theta'], steps)
                self.logger.add_scalar('Output/omega', info['omega'], steps)
                self.logger.add_scalar('Output/reward', reward, steps)
                self.logger.add_scalar('Input/target_dev', info['target_dev'], steps)
                self.logger.add_scalar('Input/action', action, steps)

            # log episodic stats
            self.logger.add_scalar('Validation/cum_reward', ep_cum_reward, ep)
            self.logger.add_scalar('Validation/ep_length', self.env_time._elapsed_steps, ep)

            # reset stats
            ep_cum_reward / steps
            ep_cum_reward = 0

            steps = 0

            # reset environment
            self.env.reset()
