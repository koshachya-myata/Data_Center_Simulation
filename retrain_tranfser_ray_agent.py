import platform
import random
import pandas as pd
from eplus import DataCenterEnv
import os
import numpy as np
import torch
from datetime import datetime

import ray
import ray.tune as tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm

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
    'verbose': 0
}
checkpoint_path = "/training_results/230714_234506_PPO_agent_experiment/PPO_DataCenterEnv_516b2_00000_0_2023-07-14_23-45-07/checkpoint_000005"

full_path = False
full_path = '/Users/akest/ray_results/PPO_DataCenterEnv_2023-07-17_16-47-40uketpqn2/checkpoint_000005'

pwd = os.getcwd()
print('PWD:', pwd)


if __name__ == "__main__":
    ray.init()
    tune.register_env("DataCenterEnv", DataCenterEnv)
    horizon = int(1 * 24 * 60 / 12) # ONE DAY horizon == 360
    nw = 1
    num_samples = 1

    tune_config = {
        'env': 'DataCenterEnv',
        'framework': 'torch',
        'env_config': env_config,
        #"vf_clip_param": 1000.0,

        "entropy_coeff": 0,
        "kl_target": 0.01,
        "clip_param": 0.2,
        "lr": 0.0005, #tune.uniform(0.001, 0.1),

        'num_workers': nw,
        "num_envs_per_worker": 1,
        'num_cpus_per_worker': 6,
        "num_gpus_per_worker": 0,

        "horizon": horizon,
        "soft_horizon": True,
        'rollout_fragment_length': 1,

        "metrics_smoothing_episodes": 1, # think about 2-3? # rolling avg
        "train_batch_size": nw * horizon, # num of steps or rollout?
        "batch_mode": "complete_episodes",
        'sgd_minibatch_size': min(horizon, 64),
        
        "disable_env_checking": True,
        "timesteps_per_iteration": horizon # * SIM_DAYS,

    }
    trainer = Algorithm.from_checkpoint(full_path if full_path else pwd + checkpoint_path)
    filename = 'PPO_agent_experiment'

    num_iterations = 5  # Specify the number of additional iterations
    for i in range(num_iterations):
        result = trainer.train()
        print(f'END ITERATION {i}/{num_iterations}')
        print(result)
        print('-' * 10)
    new_checkpoint_path = trainer.save()
    print(f'Model saved to {new_checkpoint_path}')
    ray.shutdown()

    # -123842139.41965471