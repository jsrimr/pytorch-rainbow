import torch.nn as nn
import torch.nn.functional as F        

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class DQN(nn.Module):

    def __init__(self, env):
        super(DQN, self).__init__()

        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n

        linear_init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        
        self.layers = nn.Sequential(
            linear_init_(nn.Linear(num_inputs, 128)),
            nn.ReLU(),
            linear_init_(nn.Linear(128, 128)),
            nn.ReLU(),
            linear_init_(nn.Linear(128, num_actions))
        )
    
    def forward(self, x):
        return self.layers(x)
