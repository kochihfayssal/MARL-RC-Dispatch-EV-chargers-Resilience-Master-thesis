from MA_env import CS_Env
import numpy as np
import random

random.seed(123)
np.random.seed(123)

distances = np.array([40, 30, 35, 42, 28, 33, 39, 29, 38, 40, 36, 29])
capacities = np.array([1000, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000])
travel_times = np.array([55, 43, 50, 58, 40, 48, 53, 44, 52, 55, 51, 41])
# traffic_volumes = np.array([800, 800, 800, 800, 800, 1000, 1100, 1200, 1300, 1400, 1350, 1300,
#                             1300, 1350, 1400, 1450, 1500, 1400, 1300, 1200, 1100, 1000, 900, 800])

E = CS_Env(distances=distances, capacities=capacities, travel_times=travel_times, network_size=3)
obs = E.reset()
# print(E.get_env_info())
#print(obs[0], obs[1], sep='\n')
Terminated, Truncated = False, False
records = []

# def obs_encoder(observations):
#     encoded_obs = np.zeros((2, 22))
#     for i in range(2):
#         values = list(observations[str(i)].values())
#         agent_obs = values[0]
#         for value in values[1:] :
#             agent_obs = np.hstack((agent_obs, value))
#         encoded_obs[i] = agent_obs
#     return encoded_obs
# res = obs_encoder(obs)
# print(res)
# print(res.shape)
# print("**************************************")
print(E.get_env_info())
print(E.render())
while True :
    action_1 = int(input("give the routing action for agent 1\n"))
    action_2 = int(input("give the repair action for agent 1\n"))
    action_3 = int(input("give the repair action for agent 1\n"))
    tired = int(input("done? give 1 not done? give 0 "))
    actions = np.array([action_1, action_2, action_3])
    observations, rewards, terminated, Truncated, info = E.step(actions)
    print(E.render())
    print(rewards)
    print(info['restoration_indexes'])
    print(E.steps)
    print(terminated, Truncated, sep='\n')
    if terminated or Truncated or tired:
        break



# for i in range(24):
#     actions = [E.action_space.sample() for _ in range(E.number_repair_crews)]
#     observations, reward, done, info = E.step(actions)
#     records.append((observations, reward, done, info, actions, E.busy_repair_crews, E.traveling_repair_crews))
#     if done == True : 
#         break

# print(f'the length of the episode is : {len(records)}')
# taken_actions = [records[i][4] for i in range(len(records))]
# obtained_dones = [records[i][2] for i in range(len(records))]
# obtained_rewards = [records[i][1] for i in range(len(records))]
# obtained_obs = [records[i][0] for i in range(len(records))]
# busy_repair = [records[i][5] for i in range(len(records))]
# busy_travel = [records[i][6] for i in range(len(records))]

# for i in range(len(records)) :
#     print(taken_actions[i])
#     print('*************************************')
#     print(f'location of agent1 : {obtained_obs[i][0]["current_Node"]}')
#     print('*************************************')
#     print(f'capacity of agent1 : {obtained_obs[i][0]["current_resource_capacity"]}')
#     print('*************************************')
#     print(f'location of agent2 : {obtained_obs[i][1]["current_Node"]}')
#     print('*************************************')
#     print(f'capacity of agent2 : {obtained_obs[i][1]["current_resource_capacity"]}')
#     print('*************************************')
#     print(obtained_dones[i])
#     print('*************************************')
#     print(obtained_rewards[i])
#     print('*************************************')
#     print(busy_repair[i])
#     print('*************************************')
#     print(busy_travel[i])
#     print('--------------------------------------')
#     print('--------------------------------------')


















#T = Network_Generator.Transport_Network(distances=distances, capacities=capacities, traffic_volumes=traffic_volumes, travel_times=travel_times, network_size=3)
#print(T.nodes, len(T.nodes), len(T.edges), T.edges, sep='\n')
#print(T.distances, T.traffic_volumes, T.travel_times, T.capacities, sep='\n')
#print(len(T.distances), len(T.traffic_volumes), len(T.travel_times), len(T.capacities), sep='\n')
# print(T.get_distance_edge((1,0), (2,0)))
# print(T.get_travel_time_edge((1,1), (1,2)))
# print(T.get_capacity_edge((1,1), (1,2)))
# print(T.get_distance_edge((2,1), (2,2)))
# print(T.get_travel_time_edge((2,1), (2,2)))
# print(T.get_capacity_edge((2,1), (2,2)))
# print(T.get_capacities_node((1,0)))
# print(T.get_travel_time_node((1,0)))
# print(T.get_capacities_node((1,1)))
# print(T.get_travel_time_node((1,1)))
# print(T.get_capacities_node((0,2)))
# print(T.get_travel_time_node((0,2)))
# print(T.get_capacities_node((2,2)))
# print(T.get_travel_time_node((2,2)))
# print(T.get_travel_time_node((0,1)))
# print(T.get_travel_time_node((2,0)))
