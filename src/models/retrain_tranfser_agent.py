import platform
import random
import pandas as pd
from environment import DataCenterEnv
import os
import numpy as np
import torch
from datetime import datetime
from ray.rllib.utils.schedules.constant_schedule import ConstantSchedule
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

SIM_DAYS = 366  # 366 for 1 Jan to 1 Jan

env_config = {
    "eplus_path": eplus_path,
    "weather_file": 'weather' + pwd_del + 'Moscow.epw',
    'days': SIM_DAYS,
    'timestep': 5,
    'verbose': 0
}
checkpoint_path = "/training_results/230714_234506_PPO_agent_experiment/PPO_DataCenterEnv_516b2_00000_0_2023-07-14_23-45-07/checkpoint_000005"

full_path = False


pwd = os.getcwd()
print('PWD:', pwd)


if __name__ == "__main__":
    ray.init()
    tune.register_env("DataCenterEnv", DataCenterEnv)
    horizon = int(1 * 24 * 12) # ONE DAY horizon == 288

    trainer = Algorithm.from_checkpoint(full_path if full_path else pwd + checkpoint_path)
    print('LR:', trainer.config['lr'])
    print(str(trainer.config))

    cur_lr = 0.0099
    lr_divider = 1.5
    lr_change_freq = 5

    trainer.config['lr'] = cur_lr
    trainer.config['cur_lr'] = cur_lr
    trainer.config['lr_schedule'] = [[0, cur_lr]] #ConstantSchedule(cur_lr, framework=None)
    trainer.get_policy('default_policy').cur_lr = cur_lr

    num_iterations = 20  # Specify the number of additional iterations
    checkpoint_freq = 5
    checkpoint_at_end = True
    for i in range(num_iterations):
        result = trainer.train()
        print(f'END ITERATION {i}/{num_iterations}')
        print(result)
        print('-' * 10)
        if i % checkpoint_freq == 0 and (i != 0 or checkpoint_freq == 1):
            new_checkpoint_path = trainer.save()
            print(f'Model saved, path: {new_checkpoint_path}')
            print('Last LR: ', cur_lr)

        if i % lr_change_freq == 0 and (i != 0 or lr_change_freq == 1):
            cur_lr /= lr_divider
            trainer.config['lr'] = cur_lr # -
            trainer.config['cur_lr'] = cur_lr # -
            #trainer.config['lr_schedule'] = ConstantSchedule(cur_lr, framework=None)
            trainer.config['lr_schedule'] = [[0, cur_lr]]
            trainer.get_policy('default_policy').cur_lr = cur_lr
    if checkpoint_at_end:
        new_checkpoint_path = trainer.save()
        print(f'Final model saved, path: {new_checkpoint_path}')

    ray.shutdown()
