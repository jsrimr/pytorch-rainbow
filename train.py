import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import numpy as np

from common.utils import epsilon_scheduler, beta_scheduler, update_target, print_log, load_model, save_model
from model import DQN
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

def train(env, args, writer): 
    current_model = DQN(env, args).to(args.device)
    target_model = DQN(env, args).to(args.device)

    if args.load_model and os.path.isfile(args.load_model):
        load_model(current_model, args)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
    beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha)
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)

    optimizer = optim.Adam(current_model.parameters(), lr=args.lr)

    reward_list, length_list, loss_list = [], [], []
    episode_reward = 0
    episode_length = 0

    prev_time = time.time()
    prev_frame = 1

    state = env.reset()
    for frame_idx in range(1, args.max_frames + 1):
        if args.render:
            env.render()

        if args.noisy:
            current_model.sample_noise()
            target_model.sample_noise()

        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, np.float32(done))

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
            beta = beta_by_frame(frame_idx)
            loss = compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta)
            loss_list.append(loss.item())
            writer.add_scalar("Loss", loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(current_model, target_model)

        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, reward_list, length_list, loss_list)
            reward_list.clear(), length_list.clear(), loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()
            save_model(current_model, args)

    save_model(current_model, args)


def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta=None):
    if args.prioritized_replay:
        state, action, reward, next_state, done, weights, indices = replay_buffer.sample(args.batch_size, beta)
    else:
        state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
        weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    q_values = current_model(state)
    target_next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    if args.double:
        next_q_values = current_model(next_state)
        next_actions = next_q_values.max(1)[1].unsqueeze(1)
        next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
    else:
        next_q_value = target_next_q_values.max(1)[0]

    expected_q_value = reward + args.gamma * next_q_value * (1 - done)

    if args.prioritized_replay:
        td_error = (q_value - expected_q_value.detach())
        prios = torch.abs(td_error) + 1e-5
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach())

    optimizer.zero_grad()
    loss.backward()
    if args.prioritized_replay:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss
