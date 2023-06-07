import platform
import random
import pandas as pd
from eplus import DataCenterEnv
import os
import numpy as np
import torch
from datetime import datetime
from rl.PPO.PPO import PPO

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

config = {
    "eplus_path": eplus_path,
    "weather_file": 'weather' + pwd_del + 'Moscow.epw',
    'days': 365,
    'timestep': 12
}
pwd = os.getcwd()
print('PWD:', pwd)


################################### Training ###################################
def train(max_ep_len, max_training_timesteps, load=False, action_std=0.5, action_std_decay_rate=0.05,
          min_action_std=0.05, lr_actor=0.0003, lr_critic=0.001, action_std_decay_freq=int(2.5e5)):
    # max timesteps in one episode
    # break training loop if timeteps > max_training_timesteps
    # starting std for action distribution (Multivariate Normal)
    # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    # minimum action_std (stop decay after action_std <= min_action_std)
    # action_std decay frequency (in num timesteps)

    ####### initialize environment hyperparameters ######
    env_name = "DC_agent"
    env = DataCenterEnv(config)

    has_continuous_action_space = True  # continuous action space; else discrete

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    random_seed = 42         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = pwd + pwd_del + "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + pwd_del + env_name + pwd_del
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + pwd_del + 'PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 1      #### change this to prevent overwriting weights in same env_name folder

    directory = pwd + pwd_del + "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + pwd_del + env_name + pwd_del
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    if load:
        ppo_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

if __name__ == "__main__":
    train(max_ep_len=105120, max_training_timesteps=int(3e6), load=True, action_std=0.5, action_std_decay_rate=0.01,
          min_action_std = 0.01, lr_actor=0.0001, lr_critic=0.001, action_std_decay_freq=int(2.5e5))
