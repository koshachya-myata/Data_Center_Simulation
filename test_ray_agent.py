import platform
import random
import pandas as pd
from eplus import DataCenterEnv
import os
import numpy as np
import torch
from datetime import datetime
from ray.rllib.algorithms.algorithm import Algorithm
import ray
import ray.tune as tune

os_type = platform.system()
pwd_del = '/'
if os_type == "Linux":
    eplus_path = "/usr/local/EnergyPlus-23-1-0/"
elif os_type == "nt":  # windows
    print('Windows support may not work')
    eplus_path = "C:\\EnergyPlus-23-1-0\\"
    pwd_del = '\\'
else:  # mac
    eplus_path = "/Applications/EnergyPlus-23-1-0/"

SIM_DAYS = 366 # 366 for 1 Jan to 1 Jan
env_config = {
    "eplus_path": eplus_path,
    "weather_file": 'weather' + pwd_del + 'Moscow.epw',
    'days': SIM_DAYS,
    'timestep': 12,
    'verbose': 1
}
checkpoint_path = "/training_results/230713_215523_PPO_agent_experiment/PPO_DataCenterEnv_d2e1c_00000_0_2023-07-13_21-55-24/checkpoint_000020"
full_path = False
full_path = '/Users/akest/ray_results/PPO_DataCenterEnv_2023-07-17_14-59-1318emp4hk/checkpoint_000005'
pwd = os.getcwd()
print('PWD:', pwd)

if __name__ == "__main__":
    ray.init()
    env = DataCenterEnv(env_config)
    tune.register_env("DataCenterEnv", DataCenterEnv)

    horizon = int(1 * 24 * 60 / 12)
    nw = 1

    agent = Algorithm.from_checkpoint(full_path if full_path else pwd + checkpoint_path)

    year_energy_cost = 0
    terminated = truncated = False

    obs, _ = env.reset(seed=42)
    data = []
    while not terminated and not truncated:
        action = agent.compute_single_action(obs)
        #print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        year_energy_cost += info['Facility_Total_Electricity_Demand_Rate']
        data.append(info)

    # get year energy loss
    print('Facility Total Electricity Demand Rate:', year_energy_cost)

    df = pd.DataFrame.from_records(data)
    file_name = pwd + pwd_del + 'ray_agent_data.parquet'
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)
    ray.shutdown()


