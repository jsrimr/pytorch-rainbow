import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def optimize_model(args, 
                   replay_buffer,
                   beta_schedule,
                   online_model,
                   target_model,
                   optimizer,
                   T):

    # Sample experience
    if args.prioritized_replay:
        experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(T))
        (obses_t, actions, rewards, next_obs, dones, weights, batch_idxes) = experience
    else:
        obses_t, actions, rewards, next_obs, dones = replay_buffer.sample(args.batch_size)
        weights, batch_idxes = np.ones_like(rewards), None
    
    # Decay Learning rate
    if args.lr_decay:
        _adjust_learning_rate(optimizer, max(args.lr * (args.max_timesteps - T) / args.max_timesteps, 1e-32))

    def make_tensor(arr):
        return torch.FloatTensor(arr).to(args.device)

    # Make Batches with torch.Tensor
    obs_batch = make_tensor(obses_t)
    act_batch = make_tensor(actions).unsqueeze(-1).long()
    rew_batch = make_tensor(rewards).unsqueeze(-1)
    next_obs_batch = make_tensor(next_obs)
    weights_batch = make_tensor(weights).unsqueeze(-1)
    done_mask = make_tensor(dones).unsqueeze(-1)

    # Calculate Q-Values
    q_values = online_model(obs_batch)
    next_q_values = online_model(next_obs_batch)
    next_target_q_values = target_model(next_obs_batch)

    q_value = q_values.gather(1, act_batch)

    if args.double_q:
        best_actions = next_q_values.max(1)[1].unsqueeze(-1)
        next_q_value = next_target_q_values.gather(1, best_actions)
    else:
        next_q_value = next_q_values.max(1)[0].unsqueeze(-1)

    expected_q_value = rew_batch + args.discount * next_q_value * (1. - done_mask)

    # loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
    loss = (q_value - expected_q_value.detach()).pow(2)
    weighted_loss = (weights_batch * loss).mean()

    optimizer.zero_grad()
    weighted_loss.backward()
    if args.grad_clip:
        nn.utils.clip_grad_norm_(online_model.parameters(),args.grad_norm)
    optimizer.step()

    # Update Prioritized Replay Buffer
    if args.prioritized_replay:
        td_error = (q_value - expected_q_value).data.cpu().numpy()
        new_priorities = np.abs(td_error) + args.prioritized_replay_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)

    return weighted_loss

def _adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr