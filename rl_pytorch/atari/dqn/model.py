import torch
import torch.nn as nn
import torch.nn.functional as F    

import math

import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(torch.Tensor(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(torch.Tensor(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class DQN(nn.Module):

    def __init__(self, env, dueling=False, noisy=False):
        super(DQN, self).__init__()
        self.input_shape = env.observation_space.shape
        self.dueling = dueling
        self.noisy = noisy

        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        latent_size = self.latent_size()
        self.flatten = Flatten()

        # Noisy Network
        if self.noisy:
            Linear = NoisyLinear
        else:
            Linear = nn.Linear

        self.action_layers = nn.Sequential(
            Linear(latent_size, 512),
            nn.ReLU(),
            Linear(512, num_actions)
        )

        # Dueling Network
        if self.dueling:
            self.state_layers = nn.Sequential(
                Linear(latent_size, 512),
                nn.ReLU(),
                Linear(512, 1)
            )

    def forward(self, x):
        x /= 255.
        latent = self.flatten(self.features(x))
        action_scores = self.action_layers(latent)

        if self.dueling:
            state_score = self.state_layers(latent)
            action_scores_centered = action_scores - action_scores.mean(dim=1, keepdim=True)
            out = state_score + action_scores_centered
        else:
            out = action_scores
        return out


    def latent_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def reset_noise(self):
        modules = list(self.action_layers.modules())
        if self.dueling:
            modules += list(self.state_layers.modules())

        for module in modules:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
