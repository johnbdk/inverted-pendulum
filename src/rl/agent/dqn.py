import random
from collections import namedtuple, deque
from rl.agent.base_agent import BaseAgent
from gym.wrappers.time_limit import TimeLimit
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

# define a Python namedtuple to store experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
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
                 nsteps=5000, 
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
                              nsteps=nsteps,
                              callbackfeq=callbackfeq, 
                              alpha=alpha, 
                              epsilon_start=epsilon_start,
                              epsilon_end=epsilon_end,
                              epsilon_decay_steps=epsilon_decay_steps,
                              gamma=gamma,
                              train_freq=train_freq,
                              test_freq=test_freq)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.q_network = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.target_network = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        # Initialize ReplayBuffer
        self.replay_buffer_size = buffer_size
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # start tensorboard session
        self.tb = SummaryWriter()

    def select_action(self, state):
        if np.random.uniform() < self.epsilon: # exploration
            return self.env.action_space.sample()
        else: # exploitation
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()

            return np.argmax(action_values.cpu().data.numpy())

    def train(self, replay_buffer):
        # Randomly sample a batch of experiences from the replay buffer
        batch = random.sample(replay_buffer, self.batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert the data into tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Forward pass through the network
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


    def run(self):
        obs = self.env.reset()
        self.env.render()

        # pick timelimit environment wrapper to extract elapsed steps
        env_time = self.env
        while not isinstance(env_time, TimeLimit):
            env_time = env_time.env

        # Initialize statistics
        ep = 0
        temp_ep_reward = 0
        temp_ep_theta_error = 0
        temp_max_swing = 0
        temp_theta_min = 0
        temp_theta_max = 0
        temp_ep_max_complete_steps = 0


        # Initialize the experience replay buffer
        replay_buffer = []

        for s in range(self.nsteps):
            self.epsilon = max(self.epsilon_start - s * ((self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps), self.epsilon_end)

            # Select action
            if np.random.uniform() < self.epsilon:  # exploration
                action = self.env.action_space.sample()
            else:  # exploitation
                with torch.no_grad():
                    state_tensor = torch.tensor([obs], device=self.device, dtype=torch.float32)
                    action = self.q_network(state_tensor).max(1)[1].view(1, 1).item()

            # Take step
            next_obs, reward, done, info = self.env.step(action)
            self.env.render()

            # Update stats
            temp_ep_reward += reward
            temp_theta_max = max(info['observation'][0], temp_theta_max)
            temp_theta_min = min(info['observation'][0], temp_theta_min)
            temp_max_swing = temp_theta_max - temp_theta_min
            temp_ep_theta_error += np.pi - np.abs(info['observation'][0])
            temp_ep_max_complete_steps = max(info['complete_steps'], temp_ep_max_complete_steps)

            # Store transition in replay buffer
            replay_buffer.append((obs, action, reward, next_obs, done))

            # If replay buffer is full, remove the oldest transition
            if len(replay_buffer) > self.replay_buffer_size:
                replay_buffer.pop(0)

            # Train the model if the replay buffer is large enough
            if len(replay_buffer) > self.batch_size:
                self.train(replay_buffer)

            # Update statistics
            if done:
                ep += 1

                # Log statistics
                self.tb.add_scalar('Parameters/epsilon', self.epsilon, ep)
                self.tb.add_scalar('Parameters/alpha', self.alpha, ep)
                self.tb.add_scalar('Parameters/gamma', self.gamma, ep)
                self.tb.add_scalar('Practical/cum_reward', temp_ep_reward, ep)
                self.tb.add_scalar('Practical/ep_length', env_time._elapsed_steps, ep)
                self.tb.add_scalar('Practical/cum_theta_error', temp_ep_theta_error, ep)
                self.tb.add_scalar('Practical/max_complete_steps', temp_ep_max_complete_steps, ep)
                self.tb.add_scalar('Practical/max_swing', temp_max_swing, ep)

                # print statistics
                print('\n---- Episode %d Completed ----' % (ep))
                print('reward: %.2f' % (temp_ep_reward))
                print('length: %d' % (env_time._elapsed_steps))
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

                # Reset the environment and the replay buffer
                obs = self.env.reset()
            else:
                obs = next_obs

            # Update the target network every target_update steps
            if s % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        self.tb.close()
