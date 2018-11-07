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
    batch_size = args.batch_size
    if len(replay_buffer) < args.learning_starts:
        return

    if args.prioritized_replay:
        experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(T))
        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
    else:
        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
        weights, batch_idxes = np.ones_like(rewards), None

    def make_tensor(arr):
        return torch.FloatTensor(arr).to(args.device)

    # Make Batches with torch.Tensor
    obs_batch = make_tensor(obses_t)
    act_batch = make_tensor(actions).unsqueeze(-1).long()
    rew_batch = make_tensor(rewards).unsqueeze(-1)
    obs_tp1_batch = make_tensor(obses_tp1)
    weights_batch = make_tensor(weights).unsqueeze(-1)
    done_mask = make_tensor(dones).unsqueeze(-1)

    # Q-Network / Target Q-Network evaluation
    q_t = online_model(obs_batch)
    q_tp1 = target_model(obs_tp1_batch)

    # Q scores for actions which we know were selected in the given state
    q_t_selected = q_t.gather(1, act_batch)

    # Compute estimate of best possible value starting from state at t + 1
    if args.double_q:
        best_actions_using_online = online_model(obs_tp1_batch).max(1)[1].unsqueeze(-1)
        q_tp1_best = q_tp1.gather(1, best_actions_using_online)
    else:
        q_tp1_best = q_tp1.max(1)[0].unsqueeze(-1)
    q_tp1_best_masked = ((1.0 - done_mask) * q_tp1_best).detach()

    # Compute RHS of bellman equation
    q_t_selected_target = rew_batch + args.discount * q_tp1_best_masked

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_t_selected, q_t_selected_target, reduction='none')
    weighted_loss = (weights_batch * loss).mean()

    # Optimize the model
    optimizer.zero_grad()
    weighted_loss.backward()

    if args.grad_clip:
        nn.utils.clip_grad_norm_(online_model.parameters(),args.grad_norm)
    optimizer.step()

    # Update Prioritized Replay Buffer
    if args.prioritized_replay:
        td_error = (q_t_selected - q_t_selected_target).data.cpu().numpy()
        new_priorities = np.abs(td_error) + args.prioritized_replay_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)
    
