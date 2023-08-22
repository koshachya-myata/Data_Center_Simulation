from src.models.constant_model import constant_agent
from src.models.train_agent import train_agent
from src.models.test_agent import test_agent
import os
import sys

pwd = os.getcwd()
args = sys.argv
process_arg = int(args[1]) if args[1] else 3
print('Process argument:', process_arg)
restore_path = os.path.join(pwd, 'models',
                            '230808_181425_agent_experiment/A2C_DataCenterEnv_4399f_00000_0_2023-08-08_18-14-26/checkpoint_004805')
checkpoint_path = os.path.join(pwd, 'models',
                               '230808_181425_agent_experiment/A2C_DataCenterEnv_4399f_00000_0_2023-08-08_18-14-26/checkpoint_004805')
start_iter = 4805
start_ts = 58508880

lr_step = 25

if __name__ == '__main__':
    print('MAIN')
    if process_arg == 1 or process_arg == 4:
        print('Process a constant agent')
        constant_agent()
    if process_arg == 2 or process_arg == 4:
        print('Train a ray agent')
        train_agent(restore_path, start_iter, start_ts, lr_step)
    if process_arg == 3 or process_arg == 4:
        print('Test a ray agent')
        test_agent(checkpoint_path=checkpoint_path)
