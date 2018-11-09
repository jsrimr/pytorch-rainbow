import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math

def DQN(env, args):
    if args.c51:
        if args.dueling:
            model = CategoricalDuelingDQN(env, args.noisy, args.Vmin, args.Vmax, args.num_atoms, args.batch_size)
        else:
            model = CategoricalDQN(env, args.noisy, args.Vmin, args.Vmax, args.num_atoms, args.batch_size)
    else:
        if args.dueling:
            model = DuelingDQN(env, args.noisy)
        else:
            model = DQNBase(env, args.noisy)
            
    return model


class DQNBase(nn.Module):
    """
    Basic DQN + NoisyNet

    Noisy Networks for Exploration
    https://arxiv.org/abs/1706.10295
    
    parameters
    ---------
    env         environment(openai gym)
    noisy       boolean value for NoisyNet. 
                If this is set to True, self.Linear will be NoisyLinear module
    """
    def __init__(self, env, noisy):
        super(DQNBase, self).__init__()
        
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.noisy = noisy

        if noisy:
            self.Linear = NoisyLinear
        else:
            self.Linear = nn.Linear

        self.flatten = Flatten()
        
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon or self.noisy:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_value = self.forward(state)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action
        
    def update_noisy_modules(self):
        if self.noisy:
            self.noisy_modules = [module for module in self.modules() if isinstance(module, NoisyLinear)]
    
    def sample_noise(self):
        for module in self.noisy_modules:
            module.sample_noise()

    def remove_noise(self):
        for module in self.noisy_modules:
            module.remove_noise()


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, noisy):
        super(DuelingDQN, self).__init__(env, noisy)
        
        self.advantage = self.fc

        self.value = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


class CategoricalDQN(DQNBase):
    """
    A Distributional Perspective on Reinforcement Learning
    https://arxiv.org/abs/1707.06887
    """

    def __init__(self, env, noisy, Vmin, Vmax, num_atoms, batch_size):
        super(CategoricalDQN, self).__init__(env, noisy)
    
        support = torch.linspace(Vmin, Vmax, num_atoms)
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
            .unsqueeze(1).expand(batch_size, num_atoms)

        self.register_buffer('support', support)
        self.register_buffer('offset', offset)
        self.num_atoms = num_atoms

        self.fc = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, self.num_actions * self.num_atoms),
        )

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, self.num_actions, self.num_atoms)
        return x
    
    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon or self.noisy:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_dist = self.forward(state)
                q_value = (q_dist * self.support).sum(2)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action


class CategoricalDuelingDQN(CategoricalDQN):

    def __init__(self, env, noisy, Vmin, Vmax, num_atoms, batch_size):
        super(CategoricalDuelingDQN, self).__init__(env, noisy, Vmin, Vmax, num_atoms, batch_size)
        
        self.advantage = self.fc

        self.value = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, num_atoms)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)

        advantage = self.advantage(x).view(-1, self.num_actions, self.num_atoms)
        value = self.value(x).view(-1, 1, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, self.num_actions, self.num_atoms)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'): # Only init after all params added (otherwise super().__init__() fails)
            nn.init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            nn.init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            nn.init.constant_(self.sigma_weight, self.sigma_init)
            nn.init.constant_(self.sigma_bias, self.sigma_init)
        
    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, self.bias + self.sigma_bias * self.epsilon_bias)
    
    def sample_noise(self):
        self.epsilon_weight = self.epsilon_weight.normal_()
        self.epsilon_bias = self.epsilon_bias.normal_()
    
    def remove_noise(self):
        self.epsilon_weight = self.epsilon_weight.zero_()
        self.epsilon_bias = self.epsilon_bias.zero_()
