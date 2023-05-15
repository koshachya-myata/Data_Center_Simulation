import platform
import random
import pandas as pd
from eplus import DataCenterEnv
import os
import numpy as np


os_type = platform.system()
if os_type == "Linux":
    eplus_path = "/usr/local/EnergyPlus-8-8-0/"
elif os_type == "nt":  # windows
    eplus_path = "C:\\EnergyPlus-8-8-0\\"
else:  # mac
    eplus_path = "/Applications/EnergyPlus-23-1-0/"
config = {
    "eplus_path": eplus_path,
    "weather_file": 'weather/Moscow.epw', #"weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
    'days': 142,
    'timestep': 12
}
pwd = os.getcwd()

env = DataCenterEnv(config)
env.reset()

is_sim_finised = False
print("Started simulation, taking first action.")
data = []
while not is_sim_finised:
    cooling_setpoint_coeff = 0.4375
    heating_setpoint_coeff = 0.5
    flow_rate_f_coeff = 0.5
    vent_min_indoor_temp_coeff = 0.5
    vent_max_indoor_temp_coeff = 0.5
    vent_delta_coeff = 0.25
    action = [cooling_setpoint_coeff, heating_setpoint_coeff, flow_rate_f_coeff,
            vent_min_indoor_temp_coeff, vent_max_indoor_temp_coeff, vent_delta_coeff]
    obs, reward, is_sim_finised, info = env.step(action)
    info['Zone_Air_Temperature_2'] = info['Zone_Air_Temperature'] + 0.5 - random.random()
    info['Zone_Air_Temperature_3'] = info['Zone_Air_Temperature'] + 0.5 - random.random()
    data.append(info)
print("Completed simulation.")
df = pd.DataFrame.from_records(data)
file_name = pwd + '/simulation_data.parquet'
df.to_parquet(file_name)
print('Simulation data saved to', file_name)