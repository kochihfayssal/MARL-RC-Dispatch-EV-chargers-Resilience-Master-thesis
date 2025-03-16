from typing import Tuple
import math 
import numpy as np
from Network_Generator import Transport_Network 

reward_coefficients = [5, 200]

class env:
    def __init__(self, distances, capacities, travel_times, network_size, num_RCs=3, num_CSs=6):
        self.network = Transport_Network(distances, capacities, travel_times, network_size)
        self.transport_network_size = network_size
        self.number_charging_stations = num_CSs
        self.number_repair_crews = num_RCs
        self.max_steps = 24

    def step(self, actions):
        self.steps += 1
        rewards = [0]*self.number_repair_crews
        info = {
            'restoration_indexes' : np.zeros(self.number_repair_crews)
         }
        
        actions = [int(action) for action in actions]
        for Rc_id, action in enumerate(actions):
            print(f'initial positions : {self.RCs_positions}')
            print(f'Just for testing __ see position_{Rc_id} : {self.RCs_positions[Rc_id]}')
            routing_action, repair_action = action//2, action%2
            prev_position = self.RCs_positions[Rc_id].copy()
            print(f'the value of the previous pos at index {Rc_id} : {prev_position}')
            Travel_times = np.array(self.network.get_travel_time_node(tuple(prev_position)))
            
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

            if self.busy_RCs[Rc_id] >= self.steps and routing_action!=0 :
                self.RCs_positions[Rc_id] = prev_position
                rewards[Rc_id] -= 10
            print(f'Just for testing __ see position_{Rc_id} after update : {self.RCs_positions[Rc_id]}')

            if repair_action and self.RCs_positions[Rc_id] in self.CSs_positions :
                CS_id = self.CSs_positions.index(self.RCs_positions[Rc_id])
                if self.damage_status[CS_id] == 1 and self.RCs_resources[Rc_id] >= self.CSs_repair_resources[CS_id] :
                    self.damage_status[CS_id] = 0
                    self.RCs_resources[Rc_id] -= self.CSs_repair_resources[CS_id]
                    self.busy_RCs[Rc_id] += self.CSs_repair_times[CS_id] + self.steps 
                    rewards[Rc_id] += (self.demand_loads[CS_id]/self.Total_required_load)*reward_coefficients[1]
                    info['restoration_indexes'][Rc_id] += self.demand_loads[CS_id]/self.Total_required_load
                    self.demand_loads[CS_id] = 0
                
        observations = self._get_observation() 
        terminated = all(status == 0 for status in self.damage_status)
        truncated = self.steps >= self.max_steps

        return observations, rewards, terminated, truncated, info

    def reset(self, options=None):
            self.steps = 0
            self.RCs_positions = [[1,0]]*self.number_repair_crews 
            self.CSs_positions = [[2,1], [1,1], [0,0], [0,2], [1,2], [2,2]]
            self.damage_status = [1,1,1,1,1,1]
            self.CSs_repair_resources = [18, 28, 4, 28, 8, 4]
            self.CSs_repair_times = [4, 6, 2, 6, 3, 2]
            self.RCs_resources = [56]*self.number_repair_crews 
            self.demand_loads = [132.9, 280.35, 253.59, 303.07, 203, 174.39]
            self.Total_required_load = np.sum(self.demand_loads)
            self.busy_RCs = [0]*self.number_repair_crews

            obs = self._get_observation()
            return obs

    def _get_observation(self):
        observations = {}
        for Rc_id in range(self.number_repair_crews):
            if self.busy_RCs[Rc_id] == self.steps and self.steps>0 :
                self.busy_RCs[Rc_id] = 0

            obs = {
                'current_Node' : np.array(self.RCs_positions[Rc_id]),
                'CSs_status': np.array(self.damage_status, np.int8),
                'CSs_locations' : np.array(self.CSs_positions).flatten(),
                'demand_loads' : np.array(self.demand_loads, dtype=np.float32),
                'CSs_resource_required' : np.array(self.CSs_repair_resources),
                'CSs_time_required' : np.array(self.CSs_repair_times),
                'current_resource_capacity' : self.RCs_resources[Rc_id],
                'busy_Status' : self.busy_RCs[Rc_id]
            }
            observations[str(Rc_id)] = obs
        return observations

    def render(self, mode='human'):
        obs = self._get_observation()
        string = ""
        for Rc_id in range(self.number_repair_crews):
            string += '----------------------------------------\n'
            string += f'agent_{Rc_id} position  : ' + str(obs[str(Rc_id)]['current_Node'])
            string += '----------------------------------------\n'
            string += f'agent_{Rc_id} resource  : ' + str(obs[str(Rc_id)]['current_resource_capacity'])
            string += '----------------------------------------\n'
            string += f'agent_{Rc_id} busy  : ' + str(obs[str(Rc_id)]['busy_Status'])
        string += '----------------------------------------\n'
        string += 'Stations status:\n'
        string += f'damage status : ' + str(obs["1"]['CSs_status']) + '\n'
        string += '----------------------------------------\n'
        string += f'CSs Locations : ' + str(obs["1"]['CSs_locations']) + '\n'
        string += '----------------------------------------\n'
        string += f'demand load : ' + str(obs["1"]['demand_loads']) + '\n'
        string += '----------------------------------------\n'
        string += f'CS resources : ' + str(obs["1"]['CSs_time_required'])  + '\n'
        string += '----------------------------------------\n'
        string += f'CS times : ' + str(obs["1"]['CSs_resource_required'])  + '\n'
        string += '----------------------------------------\n'

        return string

if __name__ == '__main__':
    distances = np.array([40, 30, 35, 42, 28, 33, 39, 29, 38, 40, 36, 29])
    capacities = np.array([1000, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000, 800, 1000, 1000])
    travel_times = np.array([55, 43, 50, 58, 40, 48, 53, 44, 52, 55, 51, 41])
    E = env(distances=distances, capacities=capacities, travel_times=travel_times, network_size=3)
    obs = E.reset()
    Terminated, Truncated = False, False
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
