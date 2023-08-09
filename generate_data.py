import platform
import pandas as pd
from eplus import DataCenterEnv
import os

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
    'timestep': 5,
    'verbose': 1,
}
pwd = os.getcwd()

if __name__ == "__main__":
    env = DataCenterEnv(config)
    env.reset(seed=42)

    is_sim_finised = False
    print("Started simulation, taking first action.")
    data = []

    cooling_setpoint_coeff = 0.0588235  # 0.17647 -- for SP18
    hum_set_point = 0.15  # 0.29599
    ahu_set_point = 0.5

    while not is_sim_finised:
        action = [cooling_setpoint_coeff, hum_set_point, ahu_set_point]
        obs, reward, is_sim_finised, is_turncated, info = env.step(action)
        print(info)
        data.append(info)

    print("Completed simulation.")
    df = pd.DataFrame.from_records(data)
    file_name = pwd + pwd_del + 'simulation_data_SP16.parquet'
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)
