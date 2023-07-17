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
        "clip_param": 0.3,
        "lr": 0.001, #tune.uniform(0.001, 0.1),

        'num_workers': nw,
        "num_envs_per_worker": 1,
        'num_cpus_per_worker': 5,
        "num_gpus_per_worker": 0,

        "soft_horizon": True,
        "horizon": horizon,
        'rollout_fragment_length': 1, # horizon, # or train_batch_size / nw? (maybe 1 if it episodes, not timestamps)

        "metrics_smoothing_episodes": 1, # think about 2-3? # rolling avg
        "train_batch_size": nw * horizon, # num of steps or rollout?
        "batch_mode": "complete_episodes",
        'sgd_minibatch_size': min(horizon, 64),
        
        "disable_env_checking": True,
        "timesteps_per_iteration": horizon, #* SIM_DAYS,

    }
    filename = 'PPO_agent_experiment'
    analysis = tune.run(
        'PPO',
        config=tune_config,
    
        name=datetime.now().strftime("%y%m%d_%H%M%S_") + filename,
        storage_path=os.path.join(os.path.expanduser('.'), 'training_results'),

        stop={
            'training_iteration': 3,
        }, 

        num_samples=num_samples,
        metric="episode_reward_mean",
        mode="max",

        checkpoint_freq=1,
        checkpoint_at_end=True,

        restore=None,

        verbose=3,
        fail_fast=True,#"raise",
    )

    best_trial = analysis.best_trial  # Get best trial
    best_config = analysis.best_config  # Get best trial's hyperparameters
    best_logdir = analysis.best_logdir  # Get best trial's logdir
    best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
    best_result = analysis.best_result  # Get best trial's last results
    best_result_df = analysis.best_result_df  # Get best result as pandas dataframe

    df = analysis.trial_dataframes

    print(best_trial, end='\n\n\n')
    print(best_config, end='\n\n\n')
    print(best_logdir, end='\n\n\n')


    ray.shutdown()