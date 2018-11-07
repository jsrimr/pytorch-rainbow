# -*- coding: utf-8 -*-
import argparse
import os
import gym
import torch

# TODO (Aiden)
# This is Temporary solution to import problem
# I want a better solution..

import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..'))

from RL_pytorch.common.misc_util import set_global_seeds
from RL_pytorch.cartpole.dqn.arguments import get_args
from RL_pytorch.cartpole.dqn.model import DQN
from RL_pytorch.cartpole.dqn.train import train
from RL_pytorch.cartpole.dqn.test import test

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    # Setup
    args = get_args()
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))
    set_global_seeds(args.seed)

    # Create tensorboardX writer
    if not args.evaluate:
        writer = SummaryWriter()
    
    # Create network
    env = gym.make(args.env)
    online_model = DQN(env).to(args.device)
    target_model = DQN(env).to(args.device)
    target_model.eval()

    # Load pretrained weights
    if args.model and os.path.isfile(args.model):
        online_model.load_state_dict(torch.load(args.model))

    # Copy weights from main to target model
    target_model.load_state_dict(online_model.state_dict())

    # Create optimizer for network parameters
    optimizer = torch.optim.Adam(online_model.parameters(), lr=args.lr)
    env.close()

    if not args.evaluate:
        train(args, online_model, target_model, optimizer, writer)

    test(args, online_model, writer, args.T_max)
    
    print("Done!")