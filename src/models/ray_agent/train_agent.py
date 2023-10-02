import platform
from src.dc_env.data_center_env import DataCenterEplusEnv
import os
import numpy as np
from datetime import datetime
from gymnasium import spaces

import ray
import ray.tune as tune

from src.dc_env.make_config import make_config
SIM_DAYS = 366
env_config, horizon, pwd = make_config(sim_days=SIM_DAYS)

pwd = os.getcwd()

acts_len = 3

def train_agent(restore_path=None,
                start_iter=0,
                start_ts=0,
                lr_step=25
                ):
    # TO DO: CHANGE THE LIMITS OF ACTORS ACTIONS ?
    # TEST FOR COMPLEX ENV DAYS
    ray.init()
    tune.register_env("DataCenterEnv", DataCenterEplusEnv)
    horizon = int(1 * 24 * TIMESTEP)
    nw = 1
    num_samples = 1

    tune_config = {
        'env': 'DataCenterEnv',
        'framework': 'torch',
        'env_config': env_config,
        #
        # "observation_filter": "MeanStdFilter",
        'action_space': spaces.Box(np.array([0] * acts_len), np.array([1] * acts_len), dtype=np.float32),

        # "vf_loss_coeff": 0.5,
        # "vf_clip_param": 200.0,

        "entropy_coeff": 0.15,

        'exploration_config': {
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.25,
                "final_epsilon": 0.01,
                "epsilon_timesteps": (start_ts + 26 * lr_step * SIM_DAYS * horizon),  # Timesteps over which to anneal epsilon.
            },
        # "kl_target": 0.3,
        # "clip_param": 0.8,
        "grad_clip": 15,
        # "lr": 0.0012 / 10, #tune.uniform(5e-7, 0.0015), #0.0001, #tune.uniform(0.001, 0.1),
        # Умножать не толькн на сим дейс, ведь будет меняться кол-во дней
        'lr_schedule':[[start_ts, 0.0000000005],
                       [start_ts + lr_step * SIM_DAYS * horizon, 0.0000000005],
                       [start_ts + 2 * lr_step * SIM_DAYS * horizon, 0.0000000005],
                       [start_ts + 3 * lr_step * SIM_DAYS * horizon, 0.00000000025],
                       [start_ts + 4 * lr_step * SIM_DAYS * horizon, 0.00000000025],
                       [start_ts + 5 * lr_step * SIM_DAYS * horizon, 0.00000000025],
                       [start_ts + 6 * lr_step * SIM_DAYS * horizon, 0.00000000025],
                       [start_ts + 7 * lr_step * SIM_DAYS * horizon, 0.0000000001],
                       [start_ts + 8 * lr_step * SIM_DAYS * horizon, 0.0000000001],
                       [start_ts + 9 * lr_step * SIM_DAYS * horizon, 0.0000000001],
                       [start_ts + 10 * lr_step * SIM_DAYS * horizon, 0.0000000001],
                       [start_ts + 11 * lr_step * SIM_DAYS * horizon, 0.0000000001],
                       [start_ts + 12 * lr_step * SIM_DAYS * horizon, 0.00000000005],
                       [start_ts + 13 * lr_step * SIM_DAYS * horizon, 0.00000000005],
                       [start_ts + 14 * lr_step * SIM_DAYS * horizon, 0.00000000005],
                       [start_ts + 15 * lr_step * SIM_DAYS * horizon, 0.00000000005],
                       [start_ts + 16 * lr_step * SIM_DAYS * horizon, 0.000000000025],
                       [start_ts + 17 * lr_step * SIM_DAYS * horizon, 0.000000000025],
                       [start_ts + 18 * lr_step * SIM_DAYS * horizon, 0.000000000025],
                       [start_ts + 19 * lr_step * SIM_DAYS * horizon, 0.0000000000025],
                       [start_ts + 20 * lr_step * SIM_DAYS * horizon, 0.000000000001],
                       [start_ts + 21 * lr_step * SIM_DAYS * horizon, 0.000000000001],
                       [start_ts + 22 * lr_step * SIM_DAYS * horizon, 0.000000000001],
                       [start_ts + 23 * lr_step * SIM_DAYS * horizon, 0.0000000000005],
                       [start_ts + 24 * lr_step * SIM_DAYS * horizon, 0.0000000000005],
                       [start_ts + 25 * lr_step * SIM_DAYS * horizon, 0.0000000000005],
                       ],

        'num_workers': nw,
        "num_envs_per_worker": 1,
        'num_cpus_per_worker': 6,
        "num_gpus_per_worker": 0,

        # "soft_horizon": True,
        "horizon": horizon,
        'rollout_fragment_length': horizon,  # horizon, # or train_batch_size / nw?

        "metrics_smoothing_episodes": 1,  # think about 2-3? # rolling avg
        "train_batch_size": nw * horizon,  # num of steps or rolloutы?
        "batch_mode": "complete_episodes",  # "truncate_episodes",
        'sgd_minibatch_size': min(horizon, 64),  # not for A2C

        "disable_env_checking": True,
        "timesteps_per_iteration": horizon * SIM_DAYS,  # * nw

        'use_gae': True,



        # "optimizer": { "type": "Adam", "lr": 0.0001, },

    }
    filename = 'agent_experiment'
    analysis = tune.run(
        'A2C',
        config=tune_config,
    
        name=datetime.now().strftime("%y%m%d_%H%M%S_") + filename,
        storage_path=os.path.join(pwd, 'models'),

        stop={
            'training_iteration': start_iter + 26 * lr_step,
        #    'episodes_total': start_iter + 20 * lr_step,
        }, 

        num_samples=num_samples,
        # 
        metric='episode_reward_mean',
        mode="max",

        checkpoint_freq=5,
        checkpoint_at_end=True,

        restore=restore_path,

        verbose=3,
        
        #fail_fast=True,#"raise",
    )

    best_trial = analysis.best_trial  # Get best trial
    best_config = analysis.best_config  # Get best trial's hyperparameters
    best_logdir = analysis.best_logdir  # Get best trial's logdir
    best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
    best_result = analysis.best_result  # Get best trial's last results
    best_result_df = analysis.best_result_df  # Get best result as pandas dataframe

    df = analysis.trial_dataframes

    print('Best trial:', best_trial, end='\n\n\n')
    print('Best config:', best_config, end='\n\n\n')
    print('Best logdir:', best_logdir, end='\n\n\n')

    print('-' * 20, end='\n\n')
    print('BEST CHECKPOINT:', best_checkpoint, end='\n\n')
    print('BEST RESULT:', best_result, end='\n\n')
    print('-' * 20)
    ray.shutdown()
