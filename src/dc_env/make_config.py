import platform
import os

def make_config(sim_days=366, timestamp=5):
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

    env_config = {
        "eplus_path": eplus_path,
        "weather_file": 'weather' + pwd_del + 'Moscow.epw',
        'days': sim_days,
        'timestep': timestamp,
        'verbose': 1
    }

    pwd = os.getcwd()
    horizon = int(1 * 24 * timestamp)
    return env_config, horizon, pwd
