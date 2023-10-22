from src.models.constant_model import constant_agent
from src.models.ray_agent.train_agent import train_agent
from src.models.ray_agent.test_agent import test_agent
from src.models.baseline.SB3_model import train_baseline, test_baseline
import os
import sys
from datetime import datetime
from src.inference.model_api import raise_server, get_model_prediction
from src.inference.inference_agent import simulate_agent_inference

pwd = os.getcwd()
args = sys.argv
process_arg = args[1].lower()
print('Process argument:', process_arg)
restore_path = os.path.join(pwd, 'models',
                            '230808_181425_agent_experiment/A2C_DataCenterEnv_4399f_00000_0_2023-08-08_18-14-26/checkpoint_004805')
checkpoint_path = os.path.join(pwd, 'models',
                               '230808_181425_agent_experiment/A2C_DataCenterEnv_4399f_00000_0_2023-08-08_18-14-26/checkpoint_004805')
start_iter = 4805
start_ts = 58508880
CWD = os.getcwd()
lr_step = 25

dt = datetime.now().strftime("_%y%m%d_%H%M%S")

if __name__ == '__main__':
    if 'const' in process_arg:
        print('Process a constant agent')
        if '18' in process_arg:
            constant_agent()
        if '22' in process_arg:
            constant_agent(
                cooling_setpoint=-0.17647,
                file_name='simulation_data_SP22.parquet',
                dir_path=CWD
                )
        if '25' in process_arg:
            constant_agent(
                cooling_setpoint=0.17647058823529416,
                file_name='simulation_data_SP25.parquet',
                dir_path=CWD
                )
    if process_arg == 'train_ray':
        print('=' * 10 + 'DEPRICATED' + '=' * 10)
        print('Train a ray agent')
        train_agent(restore_path, start_iter, start_ts, lr_step)
    if process_arg == 'test_ray':
        print('=' * 10 + 'DEPRICATED' + '=' * 10)
        print('Test a ray agent')
        test_agent(checkpoint_path=checkpoint_path)
    if process_arg == 'train_sb3':
        train_baseline(
            save_pth=os.path.join(pwd, 'models', 'new_PPO_baseline' + dt),
            load_pth=os.path.join(pwd, 'models', 'new_PPO_baseline_231019_133911'),
            tensorboard_log=os.path.join(pwd, 'models', 'new_PPO_tb'))
    if process_arg == 'test_sb3':
        test_baseline(
            load_pth=os.path.join(pwd, 'models', 'new_PPO_baseline_231019_133911'),
            file_name='PPO_baseline_data.parquet',
            tensorboard_log=None
        )
    if process_arg == 'raise_server' or 'server' in process_arg:
        raise_server()
    if process_arg == 'simulate_inference' or 'sim' in process_arg:
        init_obs = [19, 19, 19, 20, 18, 0.2, 0.2]  # 3T + rh + O(t) + prev[Col, Ahu]
        simulate_agent_inference(init_obs)

