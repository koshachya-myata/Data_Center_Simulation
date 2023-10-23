"""Set main variables."""
import platform

os_type = platform.system()

ENERGYPLUS_PATH = ""
PWD_DELIM = "/"
if os_type == "Linux":
    ENERGYPLUS_PATH = "/usr/local/EnergyPlus-23-1-0/energyplus"
elif os_type == "Windows":
    ENERGYPLUS_PATH = "C:\\EnergyPlusV23-1-0\\energyplus.exe"
    PWD_DELIM = "\\"
else:  # mac
    ENERGYPLUS_PATH = "/Applications/EnergyPlus-23-1-0/energyplus"

VERBOSE = 1
# next variables must be set also in .idf file
SIM_DAYS = 30
TIMESTAMPS = 5
