import platform
import pandas as pd
from eplus import DataCenterEnv
import os
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

SIM_DAYS = 366  # 366 for 1 Jan to 1 Jan
env_config = {
    "eplus_path": eplus_path,
    "weather_file": 'weather' + pwd_del + 'Moscow.epw',
    'days': SIM_DAYS,
    'timestep': 5,
    'verbose': 1


}
checkpoint_path = "/training_results/230809_020945_agent_experiment/A2C_DataCenterEnv_aae38_00000_0_2023-08-09_02-09-46/checkpoint_005455"
full_path = False

pwd = os.getcwd()
print('PWD:', pwd)

if __name__ == "__main__":
    ray.init()
    env = DataCenterEnv(env_config)
    tune.register_env("DataCenterEnv", DataCenterEnv)

    agent = Algorithm.from_checkpoint(full_path if full_path
                                      else pwd + checkpoint_path)
    agent.get_policy().config["explore"] = False
    year_energy_cost = 0
    terminated = truncated = False

    obs, _ = env.reset(seed=42)
    data = []
    while not terminated and not truncated:
        action = agent.compute_single_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        year_energy_cost += info['Facility_Total_Electricity_Demand_Rate']
        data.append(info)
    agent.stop()

    # print year energy loss
    print('Facility Total Electricity Demand Rate:', year_energy_cost)

    df = pd.DataFrame.from_records(data)
    file_name = pwd + pwd_del + 'ray_agent_data.parquet'
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)
    ray.shutdown()
