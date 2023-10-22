import platform
import pandas as pd
from src.dc_env.data_center_env import DataCenterEplusEnv
import os
from ray.rllib.algorithms.algorithm import Algorithm
import ray
import ray.tune as tune

from src.dc_env.make_config import make_config

env_config, horizon, pwd = make_config()

pwd = os.getcwd()


def test_agent(checkpoint_path, file_name='ray_agent_data.parquet'):
    ray.init()
    env = DataCenterEplusEnv(env_config)
    tune.register_env("DataCenterEnv", DataCenterEplusEnv)

    agent = Algorithm.from_checkpoint(checkpoint_path)
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
    file_name = os.path.join(pwd, 'data', file_name)
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)
    ray.shutdown()
