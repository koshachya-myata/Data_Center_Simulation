from src.dc_env.data_center_env import DataCenterEplusEnv
import os
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC
import pandas as pd
from src.dc_env.make_config import make_config

SIM_DAYS = 8
env_config, horizon, pwd = make_config(sim_days=SIM_DAYS)

def train_baseline(episodes=8000, save_pth='PPO_baseline', log_interval=3,
                   load_pth=None, tensorboard_log=None):
    env = DataCenterEplusEnv(env_config)
    # env.reset(seed=42)

    # check_env(env)

    # support_multi_env=False
    if load_pth is None:
        model = PPO(policy='MlpPolicy', env=env, verbose=1, seed=42,
                    n_steps=horizon,
                    ent_coef=.012,
                    learning_rate=3e-5,
                    use_sde=False,
                    batch_size=128,
                    tensorboard_log=tensorboard_log)
    else:
        custom_objects = {'learning_rate': 3e-6,
                          'tensorboard_log': tensorboard_log}
        model = PPO.load(load_pth, custom_objects=custom_objects)
        model.set_env(env)
    model.learn(total_timesteps=episodes * SIM_DAYS * horizon,
                log_interval=log_interval,
                progress_bar=True)
    model.save(save_pth)
    print('SAVED TO:', save_pth)
    # mean_reward, _ = model.evaluate(n_eval_episodes=1)
    # print(f"Mean reward: {mean_reward:.2f}")


def test_baseline(load_pth, file_name, tensorboard_log=None):
    env = DataCenterEplusEnv(env_config)
    custom_objects = {'tensorboard_log': tensorboard_log}
    model = PPO.load(load_pth, custom_objects=custom_objects)
    model.set_env(env)
    env.reset(seed=42)
    is_sim_finised = False

    data = []
    start_actions = [0.19, 0.29599, 0.5]
    action = start_actions[:]
    while not is_sim_finised:
        obs, reward, is_sim_finised, is_turncated, info = env.step(action)
        data.append(info)
        action, hidden_state = model.predict(observation=obs)

    print("Completed simulation.")
    file_name = os.path.join(pwd, 'data', file_name)
    df = pd.DataFrame.from_records(data)
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)
