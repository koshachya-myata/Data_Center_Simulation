"""Test development environment."""
import sys
from config import ENERGYPLUS_PATH
import os

REQUIRED_PYTHON = "python3"


def test_dev_env():
    """Test development environment."""
    system_major = sys.version_info.major

    eplus_script = ENERGYPLUS_PATH + "energyplus"
    if not os.path.isfile(eplus_script):
        FileNotFoundError('The script for energyplus 23.1.0 was not found, '
                          'path specified in config.py is: ' + eplus_script)
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes basics tests!")


if __name__ == '__main__':
    test_dev_env()
