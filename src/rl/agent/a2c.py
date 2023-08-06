# system imports
import os

# external imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# local imports
from rl.agent.base_agent import BaseAgent
from config.rl import SAVE_FREQ, ACTOR_CRITIC_PARAMS
from config.definitions import MODELS_DIR

EPOCHS = ACTOR_CRITIC_PARAMS['epochs']

class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.new_observations = []
        self.rewards = []
        self.dones = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def store(self, observation, action, new_observation, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.new_observations.append(new_observation)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self):
        convert = lambda x: [torch.from_numpy(np.array(xi)).to(torch.device(self.device), dtype=torch.float32) for xi in x]
        obs, actions, new_obs, rewards, dones = convert([self.observations, self.actions, self.new_observations, self.rewards, self.dones])
        return obs, actions, new_obs, rewards, dones

    def clear(self):
        self.observations = []
        self.actions = []
        self.new_observations = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.observations)
    
    def __repr__(self):
        obs, actions, new_obs, rewards, dones = self.get()
        repr_str = ("[BUFFER] : Observations: type:{}, shape:{}\n"
                    "[BUFFER] : Actions: type:{}, shape:{}\n"
                    "[BUFFER] : New observations: type:{}, shape:{}\n"
                    "[BUFFER] : Rewards: type:{}, shape:{}\n"
                    "[BUFFER] : Dones: type:{}, shape:{}").format(
                        type(obs), obs.shape,
                        type(actions), actions.shape,
                        type(new_obs), new_obs.shape,
                        type(rewards), rewards.shape,
                        type(dones), dones.shape)
        return repr_str

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[20, 20]):
        super(Actor, self).__init__()

        layer_sizes = [state_size] + hidden_layers
        self.fc_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)])
        
        # two outputs for each action dimension: mean, standard deviation
        self.mu = nn.Linear(layer_sizes[-1], action_size) # mu layer
        self.sigma = nn.Linear(layer_sizes[-1], action_size) # sigma layer
        
        print("Actor: ", self)
        
    def forward(self, state):
        x = state
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        
        # head
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, state_size, hidden_layers=[20, 20]):
        super(Critic, self).__init__()
        layer_sizes = [state_size] + hidden_layers + [1]
        self.fc_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)])
        print("Critic: ", self)

    def forward(self, state):
        x = state
        for layer in self.fc_layers[:-1]:
            x = F.relu(layer(x))
        return self.fc_layers[-1](x)


class A2C(BaseAgent):
    def __init__(self,
                 env,
                 callbackfeq=100,
                 gamma=0.99,
                 learning_rate=0.001,
                 alpha_entropy=0.01,
                 alpha_actor=0.01,
                 rollout_length=1000,
                 agent_refresh=1/60):

        super().__init__(env, 
                         callbackfeq,
                         alpha=learning_rate,
                         gamma=gamma, 
                         agent_refresh=agent_refresh)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define actor and critic networks
        state_size = self.env.observation_space.shape[0]
        action_size = 1
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)

        # define optimizer
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)
        
        # entropy term coefficients
        self.alpha_entropy = alpha_entropy
        self.alpha_actor = alpha_actor

        # other params
        self.rollout_length = rollout_length
        self.rb = RolloutBuffer()

    def rollout(self, curr_timesteps : int, curr_episodes: int, render : bool = False):
        # Initialize statistics
        temp_ep_reward = 0
        temp_ep_theta_error = 0
        temp_max_swing = 0
        temp_theta_min = 0
        temp_theta_max = 0
        temp_ep_max_complete_steps = 0

        with torch.no_grad():
            

            for running_step in range(self.rollout_length):
                # predict action probabilities and value
                mu, sigma, _ = self.predict(obs)

                # create a normal distribution
                # print(obs, mu, sigma, value)
                dist = torch.distributions.Normal(mu, sigma)
                
                # sample action from the distribution
                action = dist.sample()

                new_obs, reward, done, info = self.env.step(action.item())
                if render:
                    self.env.render()

                terminal = done and not info.get('TimeLimit.truncated', False)
                
                # store experiences in rollout buffer
                self.rb.store(observation=obs, 
                             action=action.item(), 
                             new_observation=new_obs, 
                             reward=reward, 
                             done=terminal)
                
                # Update stats
                temp_ep_reward += reward
                temp_theta_max = max(info['theta_bottom'], temp_theta_max)
                temp_theta_min = min(info['theta_bottom'], temp_theta_min)
                temp_max_swing = temp_theta_max - temp_theta_min
                temp_ep_theta_error += info['theta_error']
                temp_ep_max_complete_steps = max(info['complete_steps'], temp_ep_max_complete_steps)

                if done:
                    curr_episodes += 1
                    # Log statistics
                    self.logger.add_scalar('Parameters/alpha', self.alpha, curr_timesteps)
                    self.logger.add_scalar('Parameters/gamma', self.gamma, curr_timesteps)

                    self.logger.add_scalar('Practical/cum_reward', temp_ep_reward, curr_episodes)
                    self.logger.add_scalar('Practical/cum_norm_reward', temp_ep_reward/self.env_time._elapsed_steps, curr_episodes)
                    self.logger.add_scalar('Practical/ep_length', self.env_time._elapsed_steps, curr_episodes)
                    self.logger.add_scalar('Practical/cum_theta_error', temp_ep_theta_error, curr_episodes)
                    self.logger.add_scalar('Practical/max_complete_steps', temp_ep_max_complete_steps, curr_episodes)
                    self.logger.add_scalar('Practical/max_swing', temp_max_swing, curr_episodes)

                    # print statistics
                    print('\n---- Episode %d Completed ----' % (curr_episodes))
                    print('steps: %d/%d (%.2f%%)' % (curr_timesteps, self.total_timesteps, (100*curr_timesteps)/self.total_timesteps))
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

                    # Reset the environment
                    obs = self.env.reset()

                    # save model every few episodes
                    if curr_episodes % SAVE_FREQ == 0:
                        self.save()

                    # stop accumulating experiences in this rollout
                    curr_timesteps += running_step
                    break
                else:
                    obs = new_obs
        return curr_timesteps, curr_episodes
                    
    def learn(self, total_timesteps: int, callback=None, render: bool=False):
        # initialize counters
        self.total_timesteps = total_timesteps
        curr_episodes = 0; curr_timesteps = 0

        # Initialize environment
        obs = self.env.reset()
        if render:
            self.env.render()

        # train loop
        for epoch in range(EPOCHS):
            
            # retrieve experiences
            curr_timesteps, curr_episodes = self.rollout(curr_timesteps, curr_episodes, render=render)
            
            # get experiences from buffer
            buffer_obs, buffer_actions, buffer_new_obs, buffer_rewards, buffer_dones = self.rb.get()

            # calculate returns
            G = 0
            actor_losses = []
            critic_losses = []
            entropy_losses = []
            
            for t in reversed(range(len(self.rb) - 1)):
                # calculate advantage
                G = buffer_rewards[t] + self.gamma * (1 - buffer_dones[t]) * G
                mu_t, sigma_t, value_t = self.predict(buffer_obs[t])
                advantage = G - value_t.detach()

                # create distribution
                dist = torch.distributions.Normal(mu_t, sigma_t)

                # calculate log probability
                log_prob_t = dist.log_prob(buffer_actions[t])
                
                # calculate entropy
                entropy_t = dist.entropy()

                # calculate losses
                actor_loss = -(log_prob_t * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                entropy_loss = entropy_t

                # append losses
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                entropy_losses.append(entropy_loss)
            
            # total loss
            loss = sum(critic_losses) + self.alpha_actor * sum(actor_losses) + self.alpha_entropy * sum(entropy_losses)

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # clear rollout buffer
        self.rb.clear()

    def predict(self, observation, deterministic=False):
        if not torch.is_tensor(observation):
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device, dtype=torch.float32)
        mu, sigma = self.actor(observation)
        value = self.critic(observation)
        return mu, sigma, value

    def save(self, path=''):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f=os.path.join(path, 'model.pth'))

    def load(self, path):
        checkpoint = torch.load(f=os.path.join(MODELS_DIR, path, 'model.pth'))
        self.actor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
