import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--seed', type=int, default=1004,
                        help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max-timesteps', type=int, default=int(1e7), metavar='STEPS',
                        help='Number of training steps')
    parser.add_argument('--train-freq', type=int, default=4, metavar='N',
                        help='update the model every `train_freq` steps.')
    parser.add_argument('--max-episode-length', type=int, default=1000000, metavar='LENGTH',
                        help='Maximum episode length')
    parser.add_argument('--target-update', type=int, default=1000, metavar='STEPS',
                        help='Interval of target network update')
    parser.add_argument('--memory-capacity', type=int, default=10000, metavar='CAPACITY',
                        help='Maximum memory capacity')
    parser.add_argument('--model', type=str, metavar='PARAMS',
                        help='Pretrained model (state dict)')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='Environment Name')
    parser.add_argument('--episode-life', type=int, default=1,
                        help='Whether env has episode life(1) or not(0)')
    parser.add_argument('--clip-rewards', type=int, default=1,
                        help='Whether env clip rewards(1) or not(0)')
    parser.add_argument('--frame-stack', type=int, default=1,
                        help='Whether env stacks frame(1) or not(0)')
    parser.add_argument('--scale', type=int, default=0,
                        help='Whether env scales(1) or not(0)')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--double-q', action='store_true',
                        help='Enable Double-Q Learning')
    parser.add_argument('--noisy', action='store_true',
                        help='Enable Noisy Network')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='enable prioritized experience replay')
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6,
                        help='alpha value of prioritized experience replay')
    parser.add_argument('--prioritized-replay-beta', type=float, default=0.4,
                        help='beta value of prioritized experience replay')
    parser.add_argument('--prioritized-replay-eps', type=float, default=1e-6,
                        help='eps value of prioritized experience replay')
    parser.add_argument('--exploration-fraction', type=float, default=0.1, metavar='f',
                        help='fraction of entire training period over which the exploration rate is annealed')
    parser.add_argument('--exploration-final-eps', type=float, default=0.01, metavar='f',
                        help='final value of random action probability')
    parser.add_argument('--grad-clip', action='store_true',
                        help='Clip gradients with gradient norm')
    parser.add_argument('--grad-norm', type=float, default=10.0, metavar='f',
                        help='norm of gradient clipping')
    parser.add_argument('--learning-starts', type=int, default=10000, metavar='N',
                        help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--lr-decay', action='store_true',
                        help='Enable linearly decaying learning rate to 0')
    parser.add_argument('--optimizer', type=str, default="rmsprop",
                        help='Which optimizer to use (rmsprop, adam)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help="smoothing constant for RMSprop")
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='term added to the denominator to improve numerical stability for RMSprop')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='STEPS',
                        help='Number of training steps between evaluations(roughly)')
    parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N',
                        help='Number of evaluation episodes to average over')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation agent')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args
