"""Constant agent."""
from src.dc_env.data_center_env import DataCenterEplusEnv
import pandas as pd
import os
from typing import Union
from src.dc_env.make_config import make_config


def constant_agent(cooling_setpoint: float = -0.647,
                   hum_set_point: float = 0,
                   ahu_set_point: float = -0.5,
                   file_name: str = 'simulation_data_SP18.parquet',
                   dir_path: Union[None, str] = None):
    """
    Test an agent that have the constant setpoints.

    Colling: 16 = -0.88235; 18 = -0.647, 22 = -0.17647;
             25 = 0.17647058823529416
    AHU: 10 = 0, 13 = 0.5

    Args:
        cooling_setpoint (float): cooling setpoint.
        hum_set_point (float): humidity setpoint
        ahu_set_point (float): AHU setpoint
        filefile_name_path (str): name for saving data file.
        dir_path (str | None): path to directory to save file_name.
                               Defaults to None (means working directory).
    """
    env_config, horizon, cur_work_dir = make_config()
    if dir_path is None:
        dir_path = cur_work_dir

    env = DataCenterEplusEnv(env_config)
    env.reset(seed=42)

    is_sim_finised = False
    print("Started simulation, taking first action.")
    data = []

    while not is_sim_finised:
        action = [cooling_setpoint, hum_set_point, ahu_set_point]
        obs, reward, is_sim_finised, is_turncated, info = env.step(action)
        data.append(info)

    print("Completed simulation.")
    file_name = os.path.join(dir_path, 'data', file_name)
    df = pd.DataFrame.from_records(data)
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)
