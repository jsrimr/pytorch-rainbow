import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import time
from tensorboardX import SummaryWriter

from common.utils import create_log_dir, print_args, set_global_seeds
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from arguments import get_args
from train import train
from test import test

def main():
    args = get_args()
    print_args(args)

    writer = SummaryWriter(create_log_dir(args))

    env = make_atari(args.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        test(env, args)
        env.close()
        return

    train(env, args, writer)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()