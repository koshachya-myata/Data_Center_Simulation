"""Train/test stable_baselines3 model."""
from src.dc_env.data_center_env import DataCenterEplusEnv
import os
from stable_baselines3 import PPO
import pandas as pd
from src.dc_env.make_config import make_config
from config import SIM_DAYS

env_config, horizon, pwd = make_config()
n_envs = 1


def train_baseline(episodes=1000, save_pth='PPO_baseline', log_interval=3,
                   load_pth=None, tensorboard_log=None):
    """Train stable_baselines3 model."""
    env = DataCenterEplusEnv(env_config)
    # env.reset(seed=42)

    # check_env(env)

    # support_multi_env=False
    if load_pth is None:
        model = PPO(policy='MlpPolicy', env=env, verbose=1, seed=42,
                    n_steps=horizon,
                    ent_coef=.02,
                    learning_rate=5e-5,
                    use_sde=False,
                    batch_size=horizon * n_envs,
                    tensorboard_log=tensorboard_log)
    else:
        custom_objects = {'learning_rate': 1e-5,
                          'tensorboard_log': tensorboard_log}
        model = PPO.load(load_pth, custom_objects=custom_objects)
        model.set_env(env)
  
    try:
        model.learn(total_timesteps=episodes * SIM_DAYS * horizon,
                    log_interval=log_interval,
                    progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model.")
        model.save(save_pth + '_undertrained')
    else:
        model.save(save_pth)
        print('SAVED TO:', save_pth)
    # mean_reward, _ = model.evaluate(n_eval_episodes=1)
    # print(f"Mean reward: {mean_reward:.2f}")


def test_baseline(load_pth,
                  file_name='PPO_baseline_data.parquet',
                  tensorboard_log=None):
    """Test stable_baselines3 model."""
    env = DataCenterEplusEnv(env_config)
    custom_objects = {'tensorboard_log': tensorboard_log}
    model = PPO.load(load_pth, custom_objects=custom_objects)
    model.set_env(env)
    env.reset(seed=42)
    is_sim_finised = False

    data = []
    start_actions = [-0.647, -0.4, 0]
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
