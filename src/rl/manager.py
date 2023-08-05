# system imports
import os
import time

# external imports
from gym.wrappers import TimeLimit
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback

# local imports
from rl.env.custom_unbalanced_disk import CustomUnbalancedDisk, Discretizer
from rl.env.custom_pendulum import CustomPendulum
from rl.agent.q_learning import QLearning
from rl.agent.dqn import DQN
from rl.agent.a2c import A2C
from config.definitions import MODELS_DIR
from config.rl import (
    TRAIN_STEPS,
    TEST_STEPS,
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
                 env : str = 'unbalanced_disk',
                 method : str ='q_learn',
                 mode : str ='train',
                 train_steps : int = TRAIN_STEPS,
                 test_steps : int = TEST_STEPS,
                 model_path : str | None = None,
                 ) -> None:
        
        # class attributes
        self.mode = mode
        self.method = method
        self.train_steps = train_steps
        self.test_steps = test_steps

        # ----------------- env -----------------

        # define environment
        if env == 'unbalanced_disk':
            self.env = CustomUnbalancedDisk(action_space_type=ACTION_SPACE_MAP[method])
        elif env == 'pendulum':
            self.env = CustomPendulum()
        else:
            raise ValueError('Unknown env %s' % env)

        # add time limit
        self.env = TimeLimit(self.env, max_episode_steps=MAX_EPISODE_STEPS)

        # discretize if necessary
        if STATE_SPACE_MAP[method] == 'discrete':
            self.env = Discretizer(self.env, nvec=9)

        
        # ---------------- model ----------------
        # define model
        if method == 'q_learn':
            self.model = QLearning(env=self.env,
                                   callbackfeq=100,
                                   alpha=QLEARN_PARAMS['alpha'],
                                   epsilon_start=EPSILON_PARAMS['epsilon_start'],
                                   epsilon_end=EPSILON_PARAMS['epsilon_end'],
                                   epsilon_decay_steps=EPSILON_PARAMS['epsilon_decay_steps'],
                                   gamma=QLEARN_PARAMS['gamma'],
                                   agent_refresh=AGENT_REFRESH)
        elif method == 'dqn':
            self.model = DQN(env=self.env,
                              callbackfeq=100,
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
            self.model = A2C(env=self.env,
                             callbackfeq=100,
                             gamma=ACTOR_CRITIC_PARAMS['gamma'],
                             learning_rate=ACTOR_CRITIC_PARAMS['learning_rate'],
                             alpha_entropy=ACTOR_CRITIC_PARAMS['alpha_entropy'],
                             alpha_actor=ACTOR_CRITIC_PARAMS['alpha_actor'],
                             agent_refresh=AGENT_REFRESH)
        elif method == 'a2c_built':
            self.model = sb3.A2C(policy="MlpPolicy", 
                                 env=self.env,
                                 learning_rate=ACTOR_CRITIC_PARAMS['learning_rate'],
                                 n_steps=ACTOR_CRITIC_PARAMS['batch_size'],
                                 gamma=ACTOR_CRITIC_PARAMS['gamma'],
                                 ent_coef=ACTOR_CRITIC_PARAMS['alpha_entropy'],
                                 vf_coef=ACTOR_CRITIC_PARAMS['alpha_actor'],
                                 tensorboard_log=MODELS_DIR,
                                 verbose=2,
                                 device='cpu'
            )
        else:
            raise ValueError('Unknown method %s' % method)
        
        # load model
        if mode == 'test':
            assert model_path is not None
            print('Loading model %s' % model_path)
            if method == 'a2c_built':
                self.model.set_parameters(load_path_or_dict=os.path.join(MODELS_DIR, model_path + '.zip'))
            else:
                self.model.load(path=model_path)

        # reset environment
        self.init_obs = self.env.reset()

    def train(self, render=False):
        try:
            # custom callback
            cb = CustomCallback(self.env)
            
            # start training loop
            if self.method == 'a2c_built':
                self.model.learn(total_timesteps=self.train_steps, callback=cb)
            else:
                self.model.learn(total_timesteps=self.train_steps, render=render)
        finally: # ALWAYS RUN THIS
            # save model
            if self.method == 'a2c_built':
                log_dir = self.model.logger.get_dir()
            else:
                log_dir = self.model.logger.log_dir
            self.model.save(path=log_dir)

            # close environment
            self.env.close()
            
    def simulate(self):
        try:
            self.model.simulate(total_timesteps=self.test_steps)
        finally:
            self.env.close()
            
        #     # NOTE: THIS IS FOR THE STABLE_BASELINES VALIDATION (later, replace the below code with the code block above)
        #     # initialize environment
        #     obs = self.env.reset()

        #     # initialize stats
        #     ep = 0
        #     steps = 0
        #     ep_cum_reward = 0

        #     for i in range(self.test_steps):
        #         # select action
        #         action = self.model.predict(obs, deterministic=True)

        #         # step action
        #         obs, reward, done, info = self.env.step(action[0])
        #         self.env.render()
                
        #         # sleep
        #         time.sleep(AGENT_REFRESH)
                
        #         # update stats
        #         ep_cum_reward += reward
        #         steps += 1

        #         # terminal state
        #         if done:
        #             # log stats
        #             # self.logger.add_scalar('Validation/cum_reward', ep_cum_reward, ep)

        #             # reset stats
        #             ep_cum_reward / steps
        #             ep_cum_reward = 0

        #             ep += 1
        #             steps = 0

        #             # reset environment
        #             self.env.reset()
        # finally:
        #     self.env.close()
