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


# define experiences namedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42, fc1_units=24, fc2_units=24):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
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
                 train_freq=1/24,
                 test_freq=1/60,
                 buffer_size=50000,
                 batch_size=64,
                 target_update_freq=10000):
        
        super(DQN, self).__init__(env,
                              callbackfeq=callbackfeq, 
                              alpha=alpha, 
                              epsilon_start=epsilon_start,
                              epsilon_end=epsilon_end,
                              epsilon_decay_steps=epsilon_decay_steps,
                              gamma=gamma,
                              train_freq=train_freq,
                              test_freq=test_freq)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.target_network = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        # Initialize ReplayBuffer
        self.replay_buffer_size = buffer_size
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq


    def predict(self, obs, exploration=True):
        # exploration
        if exploration and np.random.uniform() < self.epsilon:  # exploration
            action = self.env.action_space.sample()
        
        # exploitation
        else:  
            with torch.no_grad():
                state_tensor = torch.tensor([obs], device=self.device, dtype=torch.float32)
                action = self.q_network(state_tensor).max(1)[1].view(1, 1).item()
        return action

    def update(self, replay_buffer):
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


    def learn(self, total_timesteps : int, callback = None, render : bool = False):
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
            temp_theta_max = max(info['theta_bottom'], temp_theta_max)
            temp_theta_min = min(info['theta_bottom'], temp_theta_min)
            temp_max_swing = temp_theta_max - temp_theta_min
            temp_ep_theta_error += info['theta_error']
            temp_ep_max_complete_steps = max(info['complete_steps'], temp_ep_max_complete_steps)

            # Terminate episode if done
            if done:
                ep += 1

                # Log statistics
                self.tb.add_scalar('Parameters/epsilon', self.epsilon, s)
                self.tb.add_scalar('Parameters/alpha', self.alpha, s)
                self.tb.add_scalar('Parameters/gamma', self.gamma, s)
                self.tb.add_scalar('Parameters/batch_size', self.batch_size, s)
                self.tb.add_scalar('Parameters/memory_len', len(self.buffer), s)
                self.tb.add_scalar('Practical/cum_reward', temp_ep_reward, ep)
                self.tb.add_scalar('Practical/cum_norm_reward', temp_ep_reward/self.env_time._elapsed_steps, ep)
                self.tb.add_scalar('Practical/ep_length', self.env_time._elapsed_steps, ep)
                self.tb.add_scalar('Practical/cum_theta_error', temp_ep_theta_error, ep)
                self.tb.add_scalar('Practical/max_complete_steps', temp_ep_max_complete_steps, ep)
                self.tb.add_scalar('Practical/max_swing', temp_max_swing, ep)
                self.tb.add_scalar('DQN/loss', temp_ep_loss, ep)

                # print statistics
                print('\n---- Episode %d Completed ----' % (ep))
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
            else:
                obs = next_obs

            # Update the target network every target_update steps
            if s % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        self.tb.close()

    def save(self):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            # 'target_network_state_dict': self.target_network.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
        }, f=os.path.join(self.log_dir, 'model.pth'))

    def load_model(self, filename):
        checkpoint = torch.load(f=os.path.join('runs', filename, 'model.pth'))
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        # self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])