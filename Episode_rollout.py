import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import random
# from env.MA_env import CS_Env

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

    def obs_encoder(self, observations):
        encoded_obs = np.zeros((self.n_agents, self.obs_shape))
        for i in range(self.n_agents):
            values = list(observations[str(i)].values())
            agent_obs = values[0]
            for value in values[1:] :
                agent_obs = np.hstack((agent_obs, value))
            encoded_obs[i] = agent_obs
        return encoded_obs

    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, ri, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], [], []
        self.env.reset()
        terminated, truncated = False, False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        while not terminated and not truncated :
            obs = self.env._get_observation()
            obs = self.obs_encoder(obs)
            o.append(obs)
            state = self.env._get_state()
            s.append(state)
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env._get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, evaluate) #TO BE RESTORED
                # action = np.random.randint(10)
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            obs, reward, terminated, truncated, info = self.env.step(actions)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            ri.append(np.reshape(info['restoration_indexes'], [self.n_agents, 1]))
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1

        o.append(self.obs_encoder(obs))
        s.append(self.env._get_state())
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env._get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            ri.append(np.zeros([self.n_agents, 1]))
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       ri=ri.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            if key == 'terminated' :
                episode[key] = np.array([episode[key]])
                episode[key] = episode[key] * 1
            else:
                episode[key] = np.array([episode[key]])

        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, step
    
# if __name__ == '__main__':
#     # random.seed(123)
#     # np.random.seed(123)
#     # torch.manual_seed(123)
#     # torch.backends.cudnn.deterministic = True
#     class args : 
#         episode_limit = 24
#         n_actions = 10
#         n_agents = 3
#         obs_shape = 34
#         state_shape = 42
#         n_episodes = 32
#         replay_dir = ''

#     arg = args()
#     distances = np.array([40, 30, 35, 42, 28, 33, 39, 29, 38, 40, 36, 29])
#     capacities = np.array([1000, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000])
#     travel_times = np.array([55, 43, 50, 58, 40, 48, 53, 44, 52, 55, 51, 41])
#     E = CS_Env(distances=distances, capacities=capacities, travel_times=travel_times, network_size=3)
#     roll = RolloutWorker(env=E, agents=None, args=arg)
#     episodes = []
#     for episode_idx in range(arg.n_episodes):
#         episode, _, steps = roll.generate_episode(episode_idx)
#         episodes.append(episode)
#     episode_batch = episodes[0]
#     episodes.pop(0)
#     for episode in episodes:
#         for key in episode_batch.keys():
#             episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
#     print(f"batch size is : {episode_batch['o'].shape[0]}")
    

#     print("obs", episode_batch['o'].shape)
#     print("actions shape", episode_batch['u'].shape)
#     print("rewards shape", episode_batch['r'].shape)
#     print("next obs shape", episode_batch['o_next'].shape)
#     print("actions hot shape", episode_batch['u_onehot'].shape)
#     print("padded shape", episode_batch['padded'].shape)
#     # print("terminated shape", episode['terminated'].shape)
#     # print(step)
#     # print(episode_reward)