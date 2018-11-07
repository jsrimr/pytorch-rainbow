import torch

from RL_pytorch.common.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch

def create_env(args):
    env_id = args.env
    kwargs = {
        "episode_life": bool(args.episode_life),
        "clip_rewards": bool(args.clip_rewards),
        "frame_stack": bool(args.frame_stack),
        "scale": bool(args.scale)
    }
    
    env = make_atari(env_id)
    env = wrap_deepmind(env, **kwargs)
    env = wrap_pytorch(env)
    return env

def obs_to_tensor(obs, args):
    if obs is not None:
        return torch.from_numpy(obs).float().unsqueeze(0).to(args.device)
    else:
        return None
