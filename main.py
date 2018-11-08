import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time
from tensorboardX import SummaryWriter

from common.utils import create_log_dir
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from arguments import get_args
from train import train
from test import test

if __name__ == "__main__":
    args = get_args()
    writer = SummaryWriter(create_log_dir(args))

    env = make_atari(args.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    train(env, args, writer)
    
    if args.evaluate:
        test(env, args)

    writer.close()
    env.close()