import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DuelingCnnDQN(nn.Module):
    def __init__(self, env):
        super(DuelingCnnDQN, self).__init__()
        
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.flatten = Flatten()
        
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(self._feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self._feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()
    
    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_value = self.forward(state)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action