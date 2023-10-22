"""Make config dict for EnergyPlus."""
from typing import Union
import os
from config import ENERGYPLUS_PATH, TIMESTAMPS, SIM_DAYS
from config import VERBOSE


def make_config(sim_days: Union[int, None] = None) -> tuple[dict[str,
                                                                 Union[str,
                                                                       float,
                                                                       int]],
                                                            int,
                                                            str]:
    """
    Construct and Return config for EnergyPlus, horizon, cwd directory.

    Args:
        sim_days (int, optional): Simulation days. Defaults to 366.

    Returns:
        tuple[dict[str, Union[str, float, int]], int, str]: env_config,
                                                            horizon, pwd.
    """
    if sim_days is None:
        sim_days = SIM_DAYS
    env_config = {
        "eplus_path": ENERGYPLUS_PATH,
        "weather_file": os.path.join('weather', 'Moscow.epw'),
        'days': sim_days,
        'timestep': TIMESTAMPS,
        'verbose': VERBOSE
    }

    pwd = os.getcwd()
    horizon = int(1 * 24 * TIMESTAMPS)
    return env_config, horizon, pwd
