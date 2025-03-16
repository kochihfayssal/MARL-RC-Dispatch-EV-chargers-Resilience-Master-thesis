import numpy as np
import random

class Damage_Scenario_Generator :
    def __init__(self, num_CSs, network_size):
        self.number_charging_stations = num_CSs
        self.network_size = network_size
        self.damage_levels = {
            'slight' : (4, 2),
            'moderate' : (8, 3),
            'extensive' : (18, 4),
            'complete' : (28, 6)
        }
        self.level_mapping = {
            0 : 'slight',
            1 : 'moderate',
            2 : 'extensive',
            3 : 'complete',
        }
    
    def generate_damage_scenario(self):
        CS_locations = []
        CS_status = []
        CS_required_resources = []
        CS_required_times = []
        for i in range(self.number_charging_stations):
            position = [np.random.randint(self.network_size), np.random.randint(1, self.network_size)]
            while position in CS_locations : # to avoid same position of RC and CS / or position in RCs_positions 
                position = [np.random.randint(self.network_size), np.random.randint(1, self.network_size)]
            CS_locations.append(position)
            level = np.random.randint(4)
            damage_level = self.level_mapping[level]
            CS_status.append(1)
            CS_required_resources.append(self.damage_levels[damage_level][0])
            CS_required_times.append(self.damage_levels[damage_level][1])
        return CS_locations, CS_status, CS_required_resources, CS_required_times


