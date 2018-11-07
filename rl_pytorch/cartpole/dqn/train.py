
import gym
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL_pytorch.common.schedule import LinearSchedule
from RL_pytorch.cartpole.dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from RL_pytorch.cartpole.dqn.optim import optimize_model
from RL_pytorch.cartpole.dqn.test import test

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
    if sample > eps_threshold:
        with torch.no_grad():
            obs = obs_to_tensor(obs, args)
            action = online_model(obs).max(1)[1].item()
    else:
        action = random.randrange(env.action_space.n)
    return action

def train(args, online_model, target_model, optimizer, writer):
    env = gym.make(args.env)
    env.seed(args.seed)
    online_model.train()

    # Initialize Exploration Scheduler
    e_schedule = LinearSchedule(int(args.T_max * args.exploration_fraction),
                                initial_p=1.0,
                                final_p=args.exploration_final_eps)

    # Initialize Replay Buffer and Beta Scheduler
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.memory_capacity,
                                                alpha=args.prioritized_replay_alpha)
        b_schedule = LinearSchedule(args.T_max,
                                    initial_p=args.prioritized_replay_beta,
                                    final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(args.memory_capacity)
        b_schedule = None

    done = True
    T = 0

    while T <= args.T_max:
        # Reset
        if done:
            obs = env.reset()
            done, episode_length = False, 0
        
        while not done:
            # Select action and Continue playing
            action = select_action(obs, e_schedule, args, env, online_model, T)
            new_obs, reward, done, _ = env.step(action)
            reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optional clip
            done = done or episode_length >= args.max_episode_length

            # Store memory to replay buffer
            replay_buffer.add(obs, action, reward, new_obs, float(done))
            obs = new_obs

            # Optimize online network and decay learning rate
            if T % args.train_freq == 0:
                optimize_model(args, replay_buffer, b_schedule, online_model, target_model, optimizer, T)
                if not args.no_lr_decay:
                    _adjust_learning_rate(optimizer, max(args.lr * (args.T_max - T) / args.T_max, 1e-32))

            # Increase step counter
            T += 1

            # Update target network
            if T % args.target_update == 0:
                target_model.load_state_dict(online_model.state_dict())

            # Test current status of online network
            if T % args.evaluation_interval == 0:
                test(args, online_model, writer, T)
    
    env.close()
