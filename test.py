import torch
import torch.optim as optim

import os
from common.utils import load_model
from model import DQN

def test(env, args): 
    current_model = DQN(env, args).to(args.device)
    current_model.eval()

    load_model(current_model, args)

    episode_reward = 0
    episode_length = 0

    state = env.reset()
    while True:
        if args.render:
            env.render()

        action = current_model.act(torch.FloatTensor(state).to(args.device), 0.)

        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            break
    
    print("Test Result - Reward {} Length {}".format(episode_reward, episode_length))
    