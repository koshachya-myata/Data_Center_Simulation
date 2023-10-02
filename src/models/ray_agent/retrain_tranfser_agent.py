import platform
import random
import pandas as pd
from src.dc_env.data_center_env import DataCenterEplusEnv
import os
import numpy as np
import torch
from datetime import datetime
from ray.rllib.utils.schedules.constant_schedule import ConstantSchedule
import ray
import ray.tune as tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm

from src.dc_env.make_config import make_config
SIM_DAYS = 366
env_config, horizon, pwd = make_config(sim_days=SIM_DAYS)

checkpoint_path = "/training_results/230714_234506_PPO_agent_experiment/PPO_DataCenterEnv_516b2_00000_0_2023-07-14_23-45-07/checkpoint_000005"

full_path = False


pwd = os.getcwd()


if __name__ == "__main__":
    ray.init()
    tune.register_env("DataCenterEnv", DataCenterEplusEnv)
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
