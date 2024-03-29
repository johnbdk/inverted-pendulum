# system imports
import random
import os
from collections import namedtuple, deque

# external imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# local imports
from rl.agent.base_agent import BaseAgent
from config.rl import SAVE_FREQ
from config.definitions import MODELS_DIR

# define experiences namedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[20, 20], seed=42):
        """
        Constructor for the QNetwork class, representing a neural network for Q-values.
        
        :param state_size: Dimension of each state.
        :param action_size: Dimension of each action.
        :param hidden_layers: List containing the number of nodes in hidden layers.
        :param seed: Random seed.
        
        Initializes the neural network with the provided parameters.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        layer_sizes = [state_size] + hidden_layers + [action_size]
        self.fc_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)])
        print(self.fc_layers)

    def forward(self, state):
        """
        Forward pass method mapping state to action values.
        
        :param state: State for which to compute action values.
        :return: Action values.
        """
        x = state
        for layer in self.fc_layers[:-1]:
            x = F.relu(layer(x))
        return self.fc_layers[-1](x)


class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Constructor for the ReplayBuffer class, representing a buffer for storing experiences.
        
        :param buffer_size: Maximum length of the buffer.
        
        Initializes the replay buffer.
        """
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """
        Method to add an experience to the buffer.
        
        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward obtained.
        :param next_state: Next state.
        :param done: Done flag.
        """
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Method to sample a batch of experiences from the buffer.
        
        :param batch_size: Size of the batch to sample.
        :return: Random sample of experiences.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Method to determine the length of the replay buffer.
        
        :return: The length of the buffer.
        """
        return len(self.buffer)


class DQN(BaseAgent):
    def __init__(self, 
                 env,
                 callbackfeq=100, 
                 alpha=0.005, 
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=0.9*5000,
                 gamma=0.99,
                 agent_refresh=1/60,
                 hidden_layers=[24, 24],
                 buffer_size=50000,
                 batch_size=64,
                 target_update_freq=10000):
        """
        Constructor for the DQN class, representing a Deep Q-Network agent.
        
        :param env: The environment to interact with.
        :param callbackfeq, alpha, gamma, agent_refresh: Parameters inherited from BaseAgent.
        :param epsilon_start, epsilon_end, epsilon_decay_steps: Parameters for epsilon-greedy policy.
        :param hidden_layers: List containing the number of nodes in hidden layers of the Q-network.
        :param buffer_size: Size of the replay buffer.
        :param batch_size: Batch size for training.
        :param target_update_freq: Frequency for updating the target network.
        
        Initializes the DQN agent with the provided parameters.
        """
        
        super(DQN, self).__init__(env,
                              callbackfeq=callbackfeq, 
                              alpha=alpha,
                              gamma=gamma,
                              agent_refresh=agent_refresh)
        
        
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_size=self.env.observation_space.shape[0], 
                                  action_size=self.env.action_space.n, 
                                  hidden_layers=hidden_layers).to(self.device)
        self.target_network = QNetwork(state_size=self.env.observation_space.shape[0], 
                                       action_size=self.env.action_space.n,
                                       hidden_layers=hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_end=epsilon_end
        self.epsilon_decay_steps=epsilon_decay_steps

        # Initialize ReplayBuffer
        self.replay_buffer_size = buffer_size
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_layers = hidden_layers

        # setup logger
        self.setup_logger()

    def learn(self, total_timesteps : int, render : bool = False):
        """
        Method to train the DQN agent.
        
        :param total_timesteps: Total number of timesteps for training.
        :param render: Flag to enable rendering. Defaults to False.
        
        Trains the agent using the DQN algorithm.
        """

        # Initialize statistics
        ep = 0
        temp_ep_reward = 0
        temp_ep_theta_error = 0
        temp_max_swing = 0
        temp_theta_min = 0
        temp_theta_max = 0
        temp_ep_max_complete_steps = 0
        temp_ep_loss = 0

        # Initialize environment
        obs = self.env.reset()
        if render:
            self.env.render()

        # Training loop
        for s in range(total_timesteps):
            # decay epsilon
            self.epsilon = max(self.epsilon_start - s * ((self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps), self.epsilon_end)

            # select action
            action = self.predict(obs)

            # apply step in environment
            next_obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render()

            # Store transition in replay buffer
            self.buffer.add(obs, action, reward, next_obs, done)

            # Train the model if the replay buffer is large enough
            if len(self.buffer) > self.batch_size:
                loss_batch = self.update(self.buffer)
                temp_ep_loss += loss_batch.item()/self.batch_size

            # Update stats
            temp_ep_reward += reward
            temp_theta_max = max(info['theta'], temp_theta_max)
            temp_theta_min = min(info['theta'], temp_theta_min)
            temp_max_swing = temp_theta_max - temp_theta_min
            temp_ep_theta_error += info['theta_error']
            temp_ep_max_complete_steps = max(info['complete_steps'], temp_ep_max_complete_steps)

            # Terminate episode if done
            if done:

                ep += 1

                # Log statistics
                self.logger.add_scalar('Parameters/epsilon', self.epsilon, s)
                self.logger.add_scalar('Parameters/alpha', self.alpha, s)
                self.logger.add_scalar('Parameters/gamma', self.gamma, s)
                self.logger.add_scalar('Parameters/batch_size', self.batch_size, s)
                self.logger.add_scalar('Parameters/memory_len', len(self.buffer), s)
                for i, neurons in enumerate(self.hidden_layers):
                    self.logger.add_scalar('Parameters/hidden_layer_' + str(i), neurons, s)

                self.logger.add_scalar('Practical/cum_reward', temp_ep_reward, s)
                self.logger.add_scalar('Practical/cum_norm_reward', temp_ep_reward/self.env_time._elapsed_steps, s)
                self.logger.add_scalar('Practical/ep_length', self.env_time._elapsed_steps, s)
                self.logger.add_scalar('Practical/cum_theta_error', temp_ep_theta_error, s)
                self.logger.add_scalar('Practical/max_complete_steps', temp_ep_max_complete_steps, s)
                self.logger.add_scalar('Practical/max_swing', temp_max_swing, s)

                self.logger.add_scalar('DQN/loss', temp_ep_loss, s)

                # print statistics
                print('\n---- Episode %d Completed ----' % (ep))
                print('steps: %d/%d (%.2f%%)' % (s, total_timesteps, (100*s)/total_timesteps))
                print('reward: %.2f' % (temp_ep_reward))
                print('length: %d' % (self.env_time._elapsed_steps))
                print('max_swing: %.2f' % (temp_max_swing))
                print('accumulated angle error: %.2f' % (temp_ep_theta_error))
                print('max complete steps: %d' % (temp_ep_max_complete_steps))
                print('---------------------------')

                # Reset statistics
                temp_ep_reward = 0
                temp_theta_min = 0
                temp_theta_max = 0
                temp_ep_theta_error = 0
                temp_ep_max_complete_steps = 0
                temp_ep_reward = 0
                temp_ep_loss = 0

                # Reset the environment
                obs = self.env.reset()

                # save model every few episodes
                if ep % SAVE_FREQ == 0:
                    self.save(path=self.get_logdir())

            else:
                obs = next_obs

            # Update the target network every target_update steps
            if s % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        # force save model at the end
        self.save(path=self.get_logdir())      

        # close summary writer
        self.logger.close()

    def predict(self, obs, deterministic=False):
        """
        Method to predict an action based on an observation.
        
        :param obs: Observation from the environment.
        :param deterministic: Flag to determine if the prediction is deterministic. Defaults to False.
        :return: Predicted action.
        
        Uses the epsilon-greedy policy for exploration and exploitation.
        """

        # exploration
        if not deterministic and np.random.uniform() < self.epsilon:  # exploration
            action = self.env.action_space.sample()
        
        # exploitation
        else:  
            with torch.no_grad():
                state_tensor = torch.tensor([obs], device=self.device, dtype=torch.float32)
                action = self.q_network(state_tensor).max(1)[1].view(1, 1).item()
        return action

    def update(self, replay_buffer):
        """
        Method to update the Q-network using experiences from the replay buffer.
        
        :param replay_buffer: Replay buffer containing experiences.
        :return: Loss value for the batch update.
        
        Performs a batch update on the Q-network using experiences sampled from the replay buffer.
        """
         
        # Randomly sample a batch of experiences from the replay buffer
        batch = replay_buffer.sample(self.batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert the data into tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Forward pass through the networkx
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        
        # Compute the target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Compute the loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def save(self, path):
        """
        Method to save the DQN model.
        
        :param path: Path to save the model.
        
        Saves the Q-network state dictionary to the specified path.
        """

        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            # 'target_network_state_dict': self.target_network.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
        }, f=os.path.join(path, 'model.pth'))

    def load(self, path):
        """
        Method to load the DQN model.
        
        :param path: Path to load the model from.
        
        Loads the Q-network state dictionary from the specified path.
        """
        
        checkpoint = torch.load(f=os.path.join(MODELS_DIR, path, 'model.pth'))
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        # self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
