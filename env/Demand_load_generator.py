from elvis.config import ScenarioConfig
from elvis.simulate import simulate
from elvis.utility.elvis_general import create_time_steps, num_time_steps
import numpy as np
import yaml
import random

class Load_Generator :
    def __init__(self, config_yaml_path, num_CSs, Damage_status):
        self.DS = Damage_status
        self.number_charging_stations = num_CSs
        self.file_path = config_yaml_path
        self.num_EVS = np.array([i*4 for i in range(1, num_CSs+1)])
    
    def generate_loads(self):
        DL = []
        for i in range(self.number_charging_stations) :
            if self.DS[i] == 1:
                num_EVs = np.random.choice(self.num_EVS)
                print(num_EVs)
                with open(self.file_path, 'r') as f :
                    yaml_str = yaml.full_load(f)
                config_from_yaml = ScenarioConfig.from_yaml(yaml_str)
                config_from_yaml.with_num_charging_events(int(num_EVs))
                # create realisation given a start and an end date and as a resolution
                realisation = config_from_yaml.create_realisation(start_date='2024-01-01 00:00:00', end_date='2024-01-07 23:00:00', resolution='01:00:00')
                results = simulate(realisation)
                start = realisation.start_date
                end = realisation.end_date
                res = realisation.resolution
                load_profile = results.aggregate_load_profile(num_time_steps(start, end, res))
                results.scenario = realisation
                DL.append(results.total_energy_charged(res))
            else :
                DL.append(0)
        return DL

# if __name__ == '__main__':
#     random.seed(123)
#     np.random.seed(123)
#     config_path = "D:/_TUBERLIN\COURSES/__Master_Thesis/Code/Environement/demand_load_config.yaml"
#     E = Load_Generator(config_path, 3, [1,1,1])
#     print(E.generate_loads())
    
