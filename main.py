import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import time

from tensorboardX import SummaryWriter

from common.utils import epsilon_scheduler, update_target, create_log_dir, print_log
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

from arguments import get_args
from model import DuelingCnnDQN
from storage import ReplayBuffer
from train import compute_td_loss

if __name__ == "__main__":
    args = get_args()
    writer = SummaryWriter(create_log_dir(args))

    env = make_atari(args.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    current_model = DuelingCnnDQN(env).to(args.device)
    target_model = DuelingCnnDQN(env).to(args.device)

    replay_buffer = ReplayBuffer(args.buffer_size)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
    optimizer = optim.Adam(current_model.parameters(), lr=args.lr)
    
    reward_list, length_list, loss_list = [], [], []
    episode_reward = 0
    episode_length = 0

    prev_time = time.time()
    prev_frame = 1

    state = env.reset()
    for frame_idx in range(1, args.max_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            state = env.reset()
            reward_list.append(episode_reward)
            length_list.append(episode_length)
            writer.add_scalar("Episode Reward", episode_reward, frame_idx)
            writer.add_scalar("Episode Length", episode_length, frame_idx)
            episode_reward, episode_length = 0, 0
        
        if len(replay_buffer) > args.learning_start:
            loss = compute_td_loss(current_model, target_model, replay_buffer, optimizer, args)
            loss_list.append(loss.item())
            writer.add_scalar("Loss", loss.item(), frame_idx)
        
        if frame_idx % args.update_target:
            update_target(current_model, target_model)
        
        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, reward_list, length_list, loss_list)
            reward_list.clear(), length_list.clear(), loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()
            