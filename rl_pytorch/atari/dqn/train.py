import gym
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from datetime import datetime
import time

from RL_pytorch.common.schedule import LinearSchedule
from RL_pytorch.atari.dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from RL_pytorch.atari.dqn.optim import optimize_model
from RL_pytorch.atari.dqn.test import test
from RL_pytorch.atari.dqn.utils import create_env, obs_to_tensor

def train(args, online_model, target_model, optimizer, writer):
    env = create_env(args)
    env.seed(args.seed)

    # Initialize Replay Buffer and Beta Scheduler
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.memory_capacity,
                                                alpha=args.prioritized_replay_alpha)
        b_schedule = LinearSchedule(args.max_timesteps,
                                    initial_p=args.prioritized_replay_beta, final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(args.memory_capacity)
        b_schedule = None

    # Initialize Exploration Scheduler
    e_schedule = LinearSchedule(int(args.max_timesteps * args.exploration_fraction),
                                initial_p=1.0, final_p=args.exploration_final_eps)

    # For logging
    all_rewards, all_lengths = [], []
    episode_reward, episode_length, episode_index, prev_T = 0, 0, 0, 0
    t0 = time.time()

    obs = env.reset()
    for T in range(1, args.max_timesteps + 1):
        if args.render:
            env.render()

        action = select_action(obs, e_schedule, args, env, online_model, T)
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, float(done))

        done = done or episode_length >= args.max_episode_length
        obs = next_obs

        episode_reward += reward
        episode_length += 1

        if done:
            obs = env.reset()
            episode_index += 1
            all_rewards.append(episode_reward)
            all_lengths.append(episode_length)
            writer.add_scalar('Reward', episode_reward, episode_index)
            writer.add_scalar('Episode Length', episode_length, episode_index)
            episode_reward, episode_length = 0, 0
        
            if episode_index % args.evaluation_interval == 0:
                name = args.save_model or args.env
                torch.save(online_model.state_dict(), '{}.pth'.format(name))
                print_log(T, prev_T, t0, all_rewards, all_lengths, episode_index, args)
                all_rewards, all_lengths = [], []
                prev_T = T
                t0 = time.time()

        if len(replay_buffer) > args.learning_starts and T % args.train_freq == 0:
            loss = optimize_model(args, replay_buffer, b_schedule, online_model, target_model, optimizer, T)
            writer.add_scalar('Loss', loss.item(), T)
        
        if args.noisy:
            online_model.reset_noise()
            target_model.reset_noise()

        if T % args.target_update == 0:
            target_model.load_state_dict(online_model.state_dict())

    env.close()

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

def print_log(T, prev_T, t0, all_rewards, all_lengths, episode_index, args):
    l = str(len(str(args.max_timesteps)))  # max num. of digits for logging steps
    print(('[{}] Step: {:<' + l + '} Episode: {:<4} FPS: {:.2f} Avg. Reward: {:.2f} Avg. Episode Length: {:.2f}').format(
        datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
        T,
        episode_index,
        float(T - prev_T) / (time.time() - t0),
        np.mean(all_rewards),
        np.mean(all_lengths)))