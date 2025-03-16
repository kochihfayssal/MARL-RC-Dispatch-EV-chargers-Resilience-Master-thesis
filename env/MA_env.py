from typing import Tuple
import math 
import numpy as np
import os
from gymnasium import Env
from gymnasium.spaces import Dict, Discrete, Box, MultiDiscrete, Tuple, MultiBinary
from .Network_Generator import Transport_Network 
from .Damage_scenario_generator import Damage_Scenario_Generator
from .Demand_load_generator import Load_Generator 


config_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(config_path, "demand_load_config.yaml")
max_resources_required = 100 
max_time_required = 100 
max_resources = 100 
reward_coefficients = [5, 10000] #the first for the travel time and the second for the load demand

class CS_Env(Env):
    def __init__(self, distances, travel_times, network_size, num_RCs=3, num_CSs=6, seed=None) :
        super(CS_Env, self).__init__()
        self.network = Transport_Network(distances, travel_times, network_size)
        self.transport_network_size = network_size
        self.number_charging_stations = num_CSs
        self.number_repair_crews = num_RCs
        self.max_steps = 24
        #self.traveling_repair_crews = np.zeros((self.number_repair_crews, 2))
        self.observation_space = Dict({
            'current_Node' : MultiDiscrete([self.transport_network_size, self.transport_network_size]),
            #'Traffic Volume' : Box(low=0, high=100, shape=(1,), dtype=np.float32),
            #'Travel_times' : Box(low=np.array([0]*4), high=np.array([500]*4), shape=(4,), dtype=np.float32),
            # 'CSs_status' : MultiBinary(self.number_charging_stations), NEW MODIF
            'CSs_locations' : MultiDiscrete([self.transport_network_size]*self.number_charging_stations*2),
            'demand_loads' : Box(low=np.array([0]*self.number_charging_stations),
                                 high=np.array([1000]*self.number_charging_stations),
                                 shape=(self.number_charging_stations,), dtype=np.float32),
            'CSs_resource_required' : MultiDiscrete([max_resources_required]*self.number_charging_stations),
            'CSs_time_required' : MultiDiscrete([max_time_required]*self.number_charging_stations),
            'current_resource_capacity' : Discrete(max_resources+1),
            'busy_Status' : Discrete(30)
            })

        self.action_space = Discrete(10) # 0:idle / 1:left / 2:right / 3:up/ 4:down
        self.seed = seed

    # Mask to remove Invalid actions --> applied to the output of the DRL model to accelerate learning performance
    def valid_actions(self, agent_id):
        i,j = self.RCs_positions[agent_id]
        possible_actions = [1, 1, 1, 1] # 0:idle / 1:left / 2:right / 3:up/ 4:down
        if j == 0:
            possible_actions[0] = 0
        if j == self.transport_network_size-1 :
            possible_actions[1] = 0
        if i == 0:
            possible_actions[2] = 0
        if i == self.transport_network_size-1 :
            possible_actions[3] = 0
        return np.array(possible_actions)
    
    # Mask to remove Invalid actions --> applied to the output of the DRL model to accelerate learning performance
    def _get_avail_agent_actions(self, agent_id):
        i,j = self.RCs_positions[agent_id]
        available_actions = [1 for _ in range(10)] 
        if j == 0:
            available_actions[2], available_actions[3] = (0,0)
        if j == self.transport_network_size-1 :
            available_actions[4], available_actions[5] = (0,0)
        if i == 0:
            available_actions[6], available_actions[7] = (0,0)
        if i == self.transport_network_size-1 :
            available_actions[8], available_actions[9] = (0,0)
        return np.array(available_actions)

    def _get_load_priorities(self, demand_loads):
        demand_loads = np.array(demand_loads)
        priorities = np.zeros(self.number_charging_stations)
        weights = np.array([1,2,4,8,16,32])
        indices = np.argsort(demand_loads)
        for i, e in enumerate(indices) : 
            priorities[e] = weights[i]
        return priorities
    
    def _find_nearest_damaged_CS(self, position):
        distances = [(cs_pos, np.linalg.norm(np.array(position) - np.array(cs_pos)))
                     for cs_pos in self.CSs_positions if cs_pos != [0, 0]]
        distances = sorted(distances, key= lambda x:x[1])
        return distances[0][0]
    
    def _distance_from_target(self, Rc_id, prev_pos, target_position):
        current_distance = np.linalg.norm(np.array(prev_pos) - np.array(target_position))
        next_distance = np.linalg.norm(np.array(self.RCs_positions[Rc_id]) - np.array(target_position))
        return next_distance < current_distance
    
    def _get_min_required_resources(self):
        R = np.array([e for e in self.CSs_repair_resources if e!=0])
        return np.min(R)

    def step(self, actions):
        self.steps += 1
        reward = 0
        info = {
            'restoration_indexes' : np.zeros(self.number_repair_crews)
         }
        
        actions = [int(action) for action in actions]
        for Rc_id, action in enumerate(actions):
            assert(self.action_space.contains(action))
            routing_action, repair_action = action//2, action%2
            prev_position = self.RCs_positions[Rc_id].copy()
            # Travel_times = np.array(self.network.get_travel_time_node(tuple(prev_position)))

            if self.busy_RCs[Rc_id] >= self.steps :
                if routing_action != 0 :
                    reward -= 10  
                continue 
            
            # penality for taking invalid actions
            possible_actions = self.valid_actions(Rc_id)
            if routing_action!=0 and possible_actions[routing_action-1] == 0 : 
                reward -= 200

            if routing_action == 1:                     
                self.RCs_positions[Rc_id][1] -= 1
            elif routing_action == 2: 
                self.RCs_positions[Rc_id][1] += 1
            elif routing_action == 3:
                self.RCs_positions[Rc_id][0] -= 1
            elif routing_action == 4:
                self.RCs_positions[Rc_id][0] += 1
            self.RCs_positions[Rc_id][0] = np.clip(self.RCs_positions[Rc_id][0], 0, self.transport_network_size-1)
            self.RCs_positions[Rc_id][1] = np.clip(self.RCs_positions[Rc_id][1], 0, self.transport_network_size-1)
            
            #penalizing for not moving when the RC is not occupied
            if self.busy_RCs[Rc_id] == 0 and routing_action == 0 and self.RCs_resources[Rc_id] != 0: 
                reward -= 200

            #penalizing movements without performing repairs. incentivize RCs to take shortest path
            if self.RCs_positions[Rc_id] != prev_position and any(E!=0 for E in self.demand_loads) :
                target_position = self._find_nearest_damaged_CS(prev_position) #added in trail_11
                if self._distance_from_target(Rc_id, prev_position, target_position) :
                    reward += 100
                else : 
                    reward -= 100 
                # road_travel_time = self.network.get_travel_time_edge(tuple(self.RCs_positions[Rc_id]), tuple(prev_position))
                # valid_indices = np.where(Travel_times != 0)[0]
                # max_direction = np.where(Travel_times == np.max(Travel_times[valid_indices]))[0][0]
                # max_road_travel_time = Travel_times[max_direction]
                # val = road_travel_time/max_road_travel_time
                # if not repair_action or not (self.RCs_positions[Rc_id] in self.CSs_positions) :
                #     reward += -road_travel_time * reward_coefficients[0]  #new modif in mappo
                                        
            if repair_action and self.RCs_positions[Rc_id] in self.CSs_positions :
                CS_id = self.CSs_positions.index(self.RCs_positions[Rc_id])
                if self.RCs_resources[Rc_id] >= self.CSs_repair_resources[CS_id] and self.demand_loads[CS_id] !=0 : #self.damage_status[CS_id] == 1 new modif
                    #self.damage_status[CS_id] = 0 new modif
                    self.RCs_resources[Rc_id] -= self.CSs_repair_resources[CS_id]
                    self.busy_RCs[Rc_id] = self.CSs_repair_times[CS_id] + self.steps 
                    reward += (self.demand_loads[CS_id]/self.Total_required_load)*reward_coefficients[1] #new modif in mappo
                    info['restoration_indexes'][Rc_id] += self.demand_loads[CS_id]/self.Total_required_load
                    self.demand_loads[CS_id] = 0
                    self.CSs_repair_resources[CS_id] = 0 #NEW MODIF added in trail_11
                    self.CSs_repair_times[CS_id] = 0 #NEW MODIF added in trail_11
                    self.CSs_positions[CS_id][0], self.CSs_positions[CS_id][1] = (0,0) #NEW MODIF added in trail_11

            #penality for unrepairing CSs for long time
            # for i in range(self.number_charging_stations) : #added in trail_9
            #     if self.damage_status[i] == 1 :
            #         rewards[Rc_id] += -0.1*self.steps
                
        observations = self._get_observation() 
        terminated = all(E == 0 for E in self.demand_loads) #new modif
        truncated = self.steps >= self.max_steps

        return observations, reward, terminated, truncated, info  # the info dictionary holds the restoration indexes to be used in critic network training

    def reset(self, options=None):
        self.steps = 0
        # depot_pos = self._random_Node()
        self.RCs_positions = [[1,0] for _ in range(self.number_repair_crews)]
        # for _ in range(self.number_repair_crews) :
        #     pos = self._random_Node()
        #     while pos in self.RCs_positions :
        #         pos = self._random_Node()
        #     self.RCs_positions.append(pos) 
        DS = Damage_Scenario_Generator(self.number_charging_stations, self.transport_network_size)
        self.CSs_positions, self.damage_status, self.CSs_repair_resources, self.CSs_repair_times = DS.generate_damage_scenario()
        DL = Load_Generator(config_path, self.number_charging_stations, self.damage_status)
        self.RCs_resources = [56 for _ in range(self.number_repair_crews)] 
        self.demand_loads = DL.generate_loads()
        # self.priorities = self._get_load_priorities(self.demand_loads)
        self.Total_required_load = np.sum(self.demand_loads)
        self.busy_RCs = [0 for _ in range(self.number_repair_crews)]

        obs = self._get_observation()
        return obs

    def _random_Node(self):
        return [np.random.randint(self.transport_network_size), np.random.randint(self.transport_network_size)]
    
    def _get_observation(self):
        observations = {}
        for Rc_id in range(self.number_repair_crews):
            if self.busy_RCs[Rc_id] == self.steps and self.steps>0 :
                self.busy_RCs[Rc_id] = 0

            obs = {
                'current_Node' : np.array(self.RCs_positions[Rc_id]),
                # 'CSs_status': np.array(self.damage_status, np.int8), NEW MODIF
                'CSs_locations' : np.array(self.CSs_positions).flatten(),
                'demand_loads' : np.array(self.demand_loads, dtype=np.float32),
                'CSs_resource_required' : np.array(self.CSs_repair_resources),
                'CSs_time_required' : np.array(self.CSs_repair_times),
                'current_resource_capacity' : self.RCs_resources[Rc_id],
                'busy_Status' : self.busy_RCs[Rc_id]
            }
            observations[str(Rc_id)] = obs
        return observations
    
    def _get_state(self):
        state = np.array(self.RCs_positions).flatten()
        state = np.hstack((state, np.array(self.CSs_positions).flatten(), np.array(self.demand_loads, dtype=np.float32),
                           np.array(self.CSs_repair_resources), np.array(self.CSs_repair_times), np.array(self.RCs_resources), np.array(self.busy_RCs)))
        return state
        
    def obs_encoder(self, observations):
        encoded_obs = np.zeros((self.n_agents, self.obs_shape))
        for i in range(self.n_agents):
            values = list(observations[str(i)].values())
            agent_obs = values[0]
            for value in values[1:] :
                agent_obs = np.hstack((agent_obs, value))
            encoded_obs[i] = agent_obs
        return encoded_obs    

    def render(self, mode='human'):
        obs = self._get_observation()
        string = ""
        for Rc_id in range(self.number_repair_crews):
            string += '----------------------------------------\n'
            string += f'agent_{Rc_id} position  : ' + str(obs[str(Rc_id)]['current_Node'])
            string += '----------------------------------------\n'
            string += f'agent_{Rc_id} valid actions : ' + str(self._get_avail_agent_actions(Rc_id))
            string += '----------------------------------------\n'
            string += f'nearest_CS : {self._find_nearest_damaged_CS(obs[str(Rc_id)]["current_Node"])}'
            string += '----------------------------------------\n'
            string += f'agent_{Rc_id} resource  : ' + str(obs[str(Rc_id)]['current_resource_capacity'])
            string += '----------------------------------------\n'
            string += f'agent_{Rc_id} busy  : ' + str(obs[str(Rc_id)]['busy_Status'])
        string += '----------------------------------------\n'
        # string += 'Stations status:\n'
        # string += f'damage status : ' + str(obs["1"]['CSs_status']) + '\n'
        # string += '----------------------------------------\n'
        string += f'CSs Locations : ' + str(obs["1"]['CSs_locations']) + '\n'
        string += '----------------------------------------\n'
        string += f'demand load : ' + str(obs["1"]['demand_loads']) + '\n'
        # string += f'load priorities : ' + str(self._get_load_priorities(self.demand_loads)) + '\n' #new modif
        string += '----------------------------------------\n'
        string += f'CS resources : ' + str(obs["1"]['CSs_resource_required'])  + '\n'
        string += '----------------------------------------\n'
        string += f'CS times : ' + str(obs["1"]['CSs_time_required'])  + '\n'
        string += '----------------------------------------\n'
        string += f'global state : {self._get_state()}' + '\n'
        string += '----------------------------------------\n'

        return string 
    
    def get_env_info(self):
        obs = self.reset()["0"]
        obs_shape = []
        for key in obs.keys() :
            if type(obs[key]) == np.ndarray :
                obs_shape.append(len(obs[key]))
            else : 
                obs_shape.append(1)
        state = self._get_state()
        return {
            "n_actions" : self.action_space.n,
            "n_agents" : self.number_repair_crews,
            "obs_shape" : np.sum(obs_shape),
            "episode_limit" : self.max_steps,
            "state_shape": state.shape[0]
        }
    
    def worst_scenario(self, options=None):
        self.steps = 0
        # depot_pos = self._random_Node()
        self.RCs_positions = [[1,0] for _ in range(self.number_repair_crews)]
        # for _ in range(self.number_repair_crews) :
        #     pos = self._random_Node()
        #     while pos in self.RCs_positions :
        #         pos = self._random_Node()
        #     self.RCs_positions.append(pos) 
        DS = Damage_Scenario_Generator(self.number_charging_stations, self.transport_network_size)
        # self.CSs_positions = [[0,1], [0,2], [1,1], [1,2], [2,1], [2,2]]
        # self.damage_status = [1,1,1,1,1,1]
        # self.CSs_repair_resources = [28 for _ in range(self.number_charging_stations)]
        # self.CSs_repair_times = [6 for _ in range(self.number_charging_stations)]
        self.CSs_positions, self.damage_status, self.CSs_repair_resources, self.CSs_repair_times = DS.generate_damage_scenario()
        DL = Load_Generator(config_path, self.number_charging_stations, self.damage_status)
        self.RCs_resources = [56 for _ in range(self.number_repair_crews)] 
        self.demand_loads = DL.generate_loads()
        # self.priorities = self._get_load_priorities(self.demand_loads)
        self.Total_required_load = np.sum(self.demand_loads)
        self.busy_RCs = [0 for _ in range(self.number_repair_crews)]

        obs = self._get_observation()
        return obs
    
    # def save_replay(self, args):
    #     out = self.render()
    #     with open(args.replay_dir + "output.txt", "w") as file:
    #         file.write(out)
