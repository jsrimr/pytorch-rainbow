import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args):
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)

    state      = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action     = torch.LongTensor(action).to(args.device)
    reward     = torch.FloatTensor(reward).to(args.device)
    done       = torch.FloatTensor(done).to(args.device)

    q_values = current_model(state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 

    if args.double:
        next_q_values = current_model(next_state)
        next_actions = next_q_values.max(1)[1].unsqueeze(1).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_actions).squeeze(1)
    else:
        next_q_value = next_q_state_values.max(1)[0]
    
    expected_q_value = reward + args.gamma * next_q_value * (1 - done)
    
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach())
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss