import gym
import random
import math
from datetime import datetime

import torch
from torch import nn

from RL_pytorch.atari.dqn.utils import create_env, obs_to_tensor

def test(args, model):
    env = create_env(args)
    env.seed(args.seed)
    model.eval()

    obs = obs_to_tensor(env.reset(), args)
    episode_reward, episode_length = 0, 0

    while True:
        if args.render:
            env.render()
        
        with torch.no_grad():
            action = model(obs).max(1)[1].item() # For memory efficiency
        
        next_obs, reward, done, _ = env.step(action)
        
        done = done or episode_length >= args.max_episode_length
        next_obs = obs_to_tensor(next_obs, args)
        obs = next_obs

        episode_reward += reward
        episode_length += 1

        if done:
            break
    
    print(('[{}] Reward: {:<8} Episode Length: {:<8}').format(
        datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
        episode_reward,
        episode_length))
    
    env.close()
    return
