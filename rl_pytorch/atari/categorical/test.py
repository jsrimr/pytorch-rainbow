import gym
import random
import math
from datetime import datetime

import torch
from torch import nn

from RL_pytorch.atari.dqn.utils import create_env

def obs_to_tensor(obs, args):
    if obs is not None:
        return torch.from_numpy(obs).float().unsqueeze(0).to(args.device)
    else:
        return None

def test(args, online_model, writer, T=None):
    env = create_env(args)
    env.seed(args.seed)
    online_model.eval()

    done = True
    t = T or 0  # Step counter for test
    l = str(len(str(args.max_timesteps)))  # max num. of digits for logging steps

    avg_rewards, avg_episode_lengths = [], []
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                obs = obs_to_tensor(env.reset(), args)
                done, episode_length = False, 0
                reward_sum = 0
            
            # Optionally render validation states
            if args.render:
                env.render()
            
            # Select action with broken graph. This is for memory efficiency
            with torch.no_grad():
                action = online_model(obs).max(1)[1].item()
            
            next_obs, reward, done, _ = env.step(action)
            
            next_obs = obs_to_tensor(next_obs, args)
            obs = next_obs

            reward_sum += reward
            done = done or episode_length >= args.max_episode_length
            episode_length += 1

            if done:
                avg_rewards.append(reward_sum)
                avg_episode_lengths.append(episode_length)
                break
    
    print(('[{}] Step: {:<' + l + '} Avg. Reward: {:<8} Avg. Episode Length: {:<8}').format(
        datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
        t,
        sum(avg_rewards) / args.evaluation_episodes,
        sum(avg_episode_lengths) / args.evaluation_episodes))
    
    if args.evaluate:
        env.close()
        return

    writer.add_scalars('Reward', {'test': sum(avg_rewards) / args.evaluation_episodes}, T)
    writer.add_scalars('Episode Length', {'test': sum(avg_episode_lengths) / args.evaluation_episodes}, T)
    
    torch.save(online_model.state_dict(), '{}.pth'.format(args.env))  # Save model params

    # Reset online model to training mode
    online_model.train()
    env.close()
