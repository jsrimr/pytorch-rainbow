
import gym
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL_pytorch.common.schedule import LinearSchedule
from RL_pytorch.atari.dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from RL_pytorch.atari.dqn.optim import optimize_model
from RL_pytorch.atari.dqn.test import test
from RL_pytorch.atari.dqn.utils import create_env

def obs_to_tensor(obs, args):
    if obs is not None:
        return torch.from_numpy(obs).float().unsqueeze(0).to(args.device)
    else:
        return None

def _adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def select_action(obs, schedule, args, env, online_model, T):
    sample = random.random()
    eps_threshold = schedule.value(T)

    # If using NoisyNet, epsilon-greedy is not used!
    if args.noisy or sample > eps_threshold:
        with torch.no_grad():
            obs = obs_to_tensor(obs, args)
            action = online_model(obs).max(1)[1].item()
    else:
        action = random.randrange(env.action_space.n)
    return action

def train(args, online_model, target_model, optimizer, writer):
    env = create_env(args)
    env.seed(args.seed)
    online_model.train()

    # Initialize Exploration Scheduler
    e_schedule = LinearSchedule(int(args.max_timesteps * args.exploration_fraction),
                                initial_p=1.0,
                                final_p=args.exploration_final_eps)

    # Initialize Replay Buffer and Beta Scheduler
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.memory_capacity,
                                                alpha=args.prioritized_replay_alpha)
        b_schedule = LinearSchedule(args.max_timesteps,
                                    initial_p=args.prioritized_replay_beta,
                                    final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(args.memory_capacity)
        b_schedule = None

    reward_sum, episode_length = 0, 0
    done = True
    T = 0

    while T <= args.max_timesteps:
        # Reset
        if done:
            obs = env.reset()
            done, episode_length = False, 0
            reward_sum = 0
        
        while not done:
            # Select action and Continue playing
            action = select_action(obs, e_schedule, args, env, online_model, T)
            new_obs, reward, done, _ = env.step(action)

            reward_sum += reward
            done = done or episode_length >= args.max_episode_length
            episode_length += 1

            # Store memory to replay buffer
            replay_buffer.add(obs, action, reward, new_obs, float(done))
            obs = new_obs

            # Optimize online network and decay learning rate
            if T % args.train_freq == 0:
                optimize_model(args, replay_buffer, b_schedule, online_model, target_model, optimizer, T)
                if args.lr_decay:
                    _adjust_learning_rate(optimizer, max(args.lr * (args.max_timesteps - T) / args.max_timesteps, 1e-32))

            # Reset noise if NoisyNet is used
            if args.noisy:
                online_model.reset_noise()
                target_model.reset_noise()

            # Increase step counter
            T += 1

            # Update target network
            if T % args.target_update == 0:
                target_model.load_state_dict(online_model.state_dict())

            # Test current status of online network
            if T % args.evaluation_interval == 0:
                test(args, online_model, writer, T)

        writer.add_scalars('Reward', {'train': reward_sum}, T)
        writer.add_scalars('Episode Length', {'train': episode_length}, T)

    env.close()
