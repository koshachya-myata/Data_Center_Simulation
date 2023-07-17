import platform
import random
import pandas as pd
from eplus import DataCenterEnv
import os
import numpy as np


os_type = platform.system()

pwd_del = '/'
if os_type == "Linux":
    eplus_path = "/usr/local/EnergyPlus-23-1-0/"
elif os_type == "nt":  # windows
    print('Windows support may not work')
    eplus_path = "C:\\EnergyPlusV23-1-0\\"
    pwd_del = '\\'
else:  # mac
    eplus_path = "/Applications/EnergyPlus-23-1-0/"
config = {
    "eplus_path": eplus_path,
    "weather_file": 'weather' + pwd_del + 'Moscow.epw',
    'days': 366,
    'timestep': 12,
    'verbose': 1,
}
pwd = os.getcwd()

if __name__ == "__main__":
    env = DataCenterEnv(config)
    env.reset(seed=42)

    is_sim_finised = False
    print("Started simulation, taking first action.")
    data = []

    cooling_setpoint_coeff = 0.135
    hum_set_point = 0.3
    ahu_set_point = 0.9
    while not is_sim_finised:
        action = [cooling_setpoint_coeff, hum_set_point, ahu_set_point]
        obs, reward, is_sim_finised, is_turncated, info = env.step(action)
        data.append(info)

    print("Completed simulation.")
    df = pd.DataFrame.from_records(data)
    file_name = pwd + pwd_del + 'simulation_data.parquet'
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)