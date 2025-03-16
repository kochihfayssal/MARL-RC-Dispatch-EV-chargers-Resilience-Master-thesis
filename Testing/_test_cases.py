import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import random
import os
from env_smac.MA_env import CS_Env
from arguments import get_mixer_args, get_common_args

def obs_encoder(observations, args):
        encoded_obs = np.zeros((args.n_agents, args.obs_shape))
        for i in range(args.n_agents):
            values = list(observations[str(i)].values())
            agent_obs = values[0]
            for value in values[1:] :
                agent_obs = np.hstack((agent_obs, value))
            encoded_obs[i] = agent_obs
        return encoded_obs

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_common_args()
    args = get_mixer_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    distances = np.array([40, 30, 35, 42, 28, 33, 39, 29, 38, 40, 36, 29])
    capacities = np.array([1000, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000])
    travel_times = np.array([55, 43, 50, 58, 40, 48, 53, 44, 52, 55, 51, 41])
    env = CS_Env(distances, capacities, travel_times, network_size=3)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.obs_shape = env_info["obs_shape"]
    args.state_shape = env_info["state_shape"]
    args.episode_limit = env_info["episode_limit"]
    for i in range(20):
        # o, u, r, ri, terminate = [], [], [], [], []
        env.reset()
        print(env.render())