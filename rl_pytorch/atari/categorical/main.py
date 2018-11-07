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
from RL_pytorch.atari.dqn.arguments import get_args
from RL_pytorch.atari.dqn.model import DQN
from RL_pytorch.atari.dqn.train import train
from RL_pytorch.atari.dqn.test import test
from RL_pytorch.atari.dqn.utils import create_env

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
    else:
        writer = None
    
    # Create network
    env = create_env(args)
    online_model = DQN(env, args.dueling).to(args.device)
    target_model = DQN(env, args.dueling).to(args.device)
    target_model.eval()

    # Load pretrained weights
    if args.model and os.path.isfile(args.model):
        # This is necessary for cpu to import model saved on gpu
        if args.device == torch.device("cpu"):
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        online_model.load_state_dict(torch.load(args.model, map_location))
            

    # Copy weights from main to target model
    target_model.load_state_dict(online_model.state_dict())

    # Create optimizer for network parameters
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(online_model.parameters(),
                                        lr=args.lr, alpha=args.alpha, eps=args.eps)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(online_model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError
    env.close()

    if not args.evaluate:
        train(args, online_model, target_model, optimizer, writer)

    test(args, online_model, writer, args.max_timesteps)
    
    print("Done!")
