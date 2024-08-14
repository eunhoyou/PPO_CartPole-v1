from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch import optim
from typing import Union

class PPO(nn.Module):
    def __init__(
        self,
        observation_space: Union[gym.spaces.Box, gym.spaces.Discrete, int],
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete, int],
        device: torch.device,
        actor_lr: float = 0.001,
        critic_lr: float = 0.005,
        n_envs: int = 4,
        clip_param: float = 0.2,
        vf_coef: float = 1.0,
        ent_coef: float = 0.01,
    ):
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.clip_param = clip_param
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        
        if isinstance(observation_space, gym.spaces.Box):
            self.obs_dim = observation_space.shape[0]
        elif isinstance(observation_space, gym.spaces.Discrete):
            self.obs_dim = observation_space.n
        elif isinstance(observation_space, int):
            self.obs_dim = observation_space
        
        if isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
            self.continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
            self.continuous = False
        elif isinstance(action_space, int):
            self.action_dim = action_space
            self.continuous = False
        
        hidden_size = 64
        actor_layers = [
            nn.Linear(self.obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        ]
        
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        
        if self.continuous:
            self.actor_mean = nn.Linear(hidden_size, self.action_dim).to(self.device)
            self.actor_logstd = nn.Linear(hidden_size, self.action_dim).to(self.device)
        else:
            self.actor_head = nn.Linear(hidden_size, self.action_dim).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        ).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])
    
    def get_action_and_value(self, state, action=None):
        actor_features = self.actor(state)
        
        if self.continuous:
            action_mean = self.actor_mean(actor_features)
            action_logstd = self.actor_logstd(actor_features)
            action_std = torch.exp(action_logstd)
            action_dist = torch.distributions.Normal(action_mean, action_std)   
            if action is None:  # old policy로 action sampling.
                action = action_dist.rsample()  # reparameterization trick         
        else:
            action_probs = torch.softmax(self.actor_head(actor_features), dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            if action is None:
                action = action_dist.sample()  # reparameterization trick
        
        if self.continuous:
            log_prob = action_dist.log_prob(action).sum(-1)
            entropy = action_dist.entropy().sum(-1)
        else:
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
        
        value = self.critic(state)
        return action, log_prob, entropy, value

    def update_parameters(self, states, actions, old_log_probs, advantages, returns):
        _, log_probs, entropy, new_values = self.get_action_and_value(states, actions)
        
        # Clipped Surrogate Objective 
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        value_loss = self.vf_coef * (new_values.squeeze() - returns).pow(2).mean()
        
        # Entropy loss
        entropy_loss = -self.ent_coef * entropy.mean()
        
        # Total loss
        total_loss = actor_loss + value_loss + entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), value_loss.item()