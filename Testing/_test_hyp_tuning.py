import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import random
import os
from env_scal.MA_env_scal import CS_Env
from arguments import get_mixer_args, get_common_args
from networks import PPOActor, TRPOActor, A2CActor, DDPGActor

def obs_encoder(observations, args):
        encoded_obs = np.zeros((args.n_agents, args.obs_shape))
        for i in range(args.n_agents):
            values = list(observations[str(i)].values())
            agent_obs = values[0]
            for value in values[1:] :
                agent_obs = np.hstack((agent_obs, value))
            encoded_obs[i] = agent_obs
        return encoded_obs

model_dir = "D:/_TUBERLIN/COURSES/__Master_Thesis/Code/Algorithms/SMAC/model/mappo/Scalability/7RCs_14_CSs_V7_H3/6_rnn_params.pkl"

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_common_args()
    args = get_mixer_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    distances = np.array([40, 30, 35, 42, 28, 33, 39, 29, 38, 40, 36, 29, 39, 28, 38, 38, 34, 41, 29, 42, 
                          38, 40, 40, 32, 30, 36, 38, 31, 39, 29, 35, 40])
    # capacities = np.array([1000, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000])
    travel_times = np.array([55, 43, 50, 58, 40, 48, 53, 44, 52, 55, 51, 41, 43, 54, 40, 52, 49, 56, 42, 58, 
                             52, 55, 47, 45, 51, 40, 52, 46, 53, 44, 50, 55])
    env = CS_Env(distances, travel_times, V_size=7, H_size=3)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.obs_shape = env_info["obs_shape"]
    args.state_shape = env_info["state_shape"]
    args.episode_limit = env_info["episode_limit"]
    model_shape = args.obs_shape + args.n_actions + args.n_agents
    actor_model = PPOActor(model_shape, args)
    actor_model.load_state_dict(torch.load(model_dir))
    actor_model.eval()
    model_rewards = []
    model_times = []
    model_loads = []
    for i in range(20):
        # o, u, r, ri, terminate = [], [], [], [], []
        env.reset()
        terminated, truncated = False, False
        episode_reward_agents = 0  # cumulative rewards
        step = 0
        model_load = 0
        last_action = np.zeros((args.n_agents, args.n_actions))
        policy_hidden = torch.zeros((1, args.n_agents, args.rnn_hidden_dim))
        while not terminated and not truncated:
            obs = env._get_observation()
            obs = obs_encoder(obs, args)
            # o.append(obs)
            actions = []
            for RC_id in range(args.n_agents):
                inputs = obs[RC_id]
                avail_actions = env._get_avail_agent_actions(RC_id)
                # mean = inputs.mean(dim=1, keepdim=True)
                # var = inputs.var(dim=1, unbiased=False, keepdim=True)
                # inputs = (inputs - mean) / torch.sqrt(var.clamp(min=1e-5))
                agent_id = np.zeros(args.n_agents)
                agent_id[RC_id] = 1.
                if args.last_action:
                    inputs = np.hstack((inputs, last_action[RC_id]))
                if args.reuse_network:
                    inputs = np.hstack((inputs, agent_id))
                inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
                avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
                # print(inputs.shape)
                with torch.no_grad():
                    outputs, policy_hidden[:, RC_id, :] = actor_model(inputs, policy_hidden[:, RC_id, :])
                action_prob = torch.nn.functional.softmax(outputs, dim=-1)
                action_prob[avail_actions == 0.0] = 0.0
                action = torch.argmax(action_prob) 
                actions.append(action)
                action_onehot = np.zeros(args.n_actions)
                action_onehot[action] = 1
                last_action[RC_id] = action_onehot
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward_agents += reward
            step += 1
            for RC_id in range(args.n_agents):
                model_load += info["restoration_indexes"][RC_id] 
        model_rewards.append(episode_reward_agents)
        model_times.append(step)
        model_loads.append(model_load)
    print(model_rewards, sum(model_rewards)/len(model_rewards), sep="\n")
    print(model_times, sum(model_times)/len(model_times), sep="\n")
    print(model_loads, sum(model_loads)/len(model_loads), sep="\n")

