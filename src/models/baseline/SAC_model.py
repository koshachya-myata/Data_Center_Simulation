import platform
from src.dc_env.data_center_env import DataCenterEnv
import os
import numpy as np
from datetime import datetime
from gymnasium import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
import pandas as pd

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

SIM_DAYS = 6
#    366 for 1 Jan to 1 Jan
TIMESTEP = 5
env_config = {
    "eplus_path": eplus_path,
    "weather_file": 'weather' + pwd_del + 'Moscow.epw',
    'days': SIM_DAYS,
    'timestep': TIMESTEP,
    'verbose': 1
}

pwd = os.getcwd()
print('PWD:', pwd)
horizon = int(1 * 24 * TIMESTEP)


def train_baseline(episodes=2200, save_pth='SAC_baseline', log_interval=3,
                   load_pth=None, tensorboard_log=None):
    env = DataCenterEnv(env_config)
    # env.reset(seed=42)

    # check_env(env)

    # support_multi_env=False
    if load_pth is None:
        model = SAC(policy='MlpPolicy', env=env, verbose=1, seed=42,
                    learning_rate=1.5e-4,
                    use_sde=False,
                    tensorboard_log=tensorboard_log)
    else:
        custom_objects = {'learning_rate': 4e-5,
                          'tensorboard_log': tensorboard_log}
        model = SAC.load(load_pth, custom_objects=custom_objects)
        model.set_env(env)
    model.learn(total_timesteps=episodes * SIM_DAYS * horizon,
                log_interval=log_interval,
                progress_bar=True)
    model.save(save_pth)
    print('SAVED TO:', save_pth)
    # mean_reward, _ = model.evaluate(n_eval_episodes=1)
    # print(f"Mean reward: {mean_reward:.2f}")


def test_baseline(load_pth, file_name, tensorboard_log=None):
    env = DataCenterEnv(env_config)
    custom_objects = {'tensorboard_log': tensorboard_log}
    model = SAC.load(load_pth, custom_objects=custom_objects)
    model.set_env(env)
    env.reset(seed=42)
    is_sim_finised = False

    data = []
    action = [0.19, 0.29599, 0.5]
    while not is_sim_finised:
        obs, reward, is_sim_finised, is_turncated, info = env.step(action)
        data.append(info)
        action, hidden_state = model.predict(observation=obs)

    print("Completed simulation.")
    file_name = os.path.join(pwd, 'data', file_name)
    df = pd.DataFrame.from_records(data)
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)
