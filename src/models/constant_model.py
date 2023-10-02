from src.dc_env.data_center_env import DataCenterEplusEnv
import platform
import pandas as pd
import os

from src.dc_env.make_config import make_config
SIM_DAYS = 366
env_config, horizon, pwd = make_config(sim_days=SIM_DAYS)


def constant_agent(cooling_setpoint_coeff=0.17647,
                   hum_set_point=0.29599,
                   ahu_set_point=0.5,
                   file_name='simulation_data_SP18.parquet'):
    """_summary_

    Args:
        cooling_setpoint_coeff (float): 0.17647 -- for SP18, 0.0588235 for SP16
        hum_set_point (float): _description_
        ahu_set_point (float): _description_
        file_path (str): _description_
    """

    env = DataCenterEplusEnv(env_config)
    env.reset(seed=42)

    is_sim_finised = False
    print("Started simulation, taking first action.")
    data = []

    while not is_sim_finised:
        action = [cooling_setpoint_coeff, hum_set_point, ahu_set_point]
        obs, reward, is_sim_finised, is_turncated, info = env.step(action)
        data.append(info)

    print("Completed simulation.")
    file_name = os.path.join(pwd, 'data', file_name)
    df = pd.DataFrame.from_records(data)
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)
