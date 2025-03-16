from runner import Runner
from env.MA_env import CS_Env
from arguments import get_mixer_args, get_common_args
import os
import numpy as np
import random
import torch

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_common_args()
    args = get_mixer_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    #distances data to be added
    distances = np.array([40, 30, 35, 42, 28, 33, 39, 29, 38, 40, 36, 29])
    #Travel times data to be added
    travel_times = np.array([55, 43, 50, 58, 40, 48, 53, 44, 52, 55, 51, 41])
    #network size to be defined dependent on the chosen transportation configuration
    env = CS_Env(distances=distances, travel_times=travel_times, network_size=3)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    runner = Runner(env, args)
    if args.learn:
        runner.run()
    