import torch
import torch.nn as nn
import torch.nn.functional as F    

import math

import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

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

        linear_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        
        conv_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        
        self.features = nn.Sequential(
            conv_init_(nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            conv_init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            conv_init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
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
            linear_init_(Linear(latent_size, 512)),
            nn.ReLU(),
            linear_init_(Linear(512, num_actions))
        )

        # Dueling Network
        if self.dueling:
            self.state_layers = nn.Sequential(
                linear_init_(Linear(latent_size, 512)),
                nn.ReLU(),
                linear_init_(Linear(512, 1))
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

class CategoricalDQN(nn.Module):
    def __init__(self, env, num_atoms, Vmin, Vmax, noisy=False):
        super(CategoricalDQN, self).__init__()

        self.num_inputs = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.noisy = noisy

        self.num_inputs = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        linear_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        
        conv_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        
        self.features = nn.Sequential(
            conv_init_(nn.Conv2d(self.num_inputs, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            conv_init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            conv_init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
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
            linear_init_(Linear(latent_size, 512)),
            nn.ReLU(),
            linear_init_(Linear(512, self.num_actions * self.num_atoms))
        )
    
    def forward(self, x):
        x /= 255.
        latent = self.flatten(self.features(x))
        action_scores = self.action_layers(latent)
        action_scores = F.softmax(action_scores.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        return action_scores
    
    def reset_noise(self):
        modules = list(self.action_layers.modules())
        for module in modules:
            if isinstance(module, NoisyLinear):
                module.reset_noise()

def projection_distribution(next_state, rewards, dones, args, target_model):
    batch_size = args.batch_size

    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms  - 1)
    support = torch.linspace(args.Vmin, args.Vmax, args.num_atoms)

    next_dist = target_model(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(1))
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = rewards + (1 - dones) * args.discount + support
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b = (Tz - args.Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * args.num_atoms, batch_size).long()\
        .unsqueeze(1).expand(batch_size, args.num_atoms)
    
    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return proj_dist