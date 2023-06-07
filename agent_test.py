import os
import time
from datetime import datetime
import torch
import numpy as np
import platform
import pandas as pd

from eplus import DataCenterEnv
from rl.PPO.PPO import PPO

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
    'days': 365,
    'timestep': 12
}

pwd = os.getcwd()

print('PWD:', pwd)

def test(max_ep_len=105120, action_std=0.05, lr_actor=0.00003,
         lr_critic = 0.0001):

    ################## hyperparameters ##################

    env_name = "DC_agent"
    env = DataCenterEnv(config)

    has_continuous_action_space = True

    total_test_episodes = 1    # total num of testing episodes

    K_epochs = 30               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    #####################################################

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 42           #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 1      #### set this to load a particular checkpoint num

    directory = pwd + pwd_del + "PPO_preTrained" + pwd_del + env_name + pwd_del
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    data = []
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward

            data.append(info)
            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")

    df = pd.DataFrame.from_records(data)
    file_name = pwd + pwd_del + 'agent_data.parquet'
    df.to_parquet(file_name)
    print('Simulation data saved to', file_name)

if __name__ == '__main__':
    test(max_ep_len=105120, action_std=0.0001, lr_actor=0.000002,
         lr_critic=0.00001)
