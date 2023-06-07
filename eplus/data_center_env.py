import os
import socket

import gymnasium as gym
import numpy as np
from eplus import Eplus_contoller
from eplus.socket_builder import socket_builder
from gymnasium import spaces
from eplus.eplus_packet_contoller import encode_packet_simple,decode_packet_simple
import platform

class DataCenterEnv(gym.Env):
    def __init__(self, config):
        os_type = platform.system()
        pwd_del = '/'
        if os_type == "nt":  # windows
            pwd_del = '\\'

        cur_dir = os.path.dirname(__file__)
        self.idf_file = cur_dir + pwd_del + "buildings" + pwd_del + "MultiZone_DC_wHot_nCold_Aisles" + pwd_del + "MultiZone_DC.idf"

        self.weather_file = cur_dir + pwd_del + config["weather_file"]

        if "eplus_path" in config:
            self.eplus_path = config["eplus_path"]
        else:
            self.eplus_path = "/Applications/EnergyPlus-23-1-0/"

        # EnergyPlus number of timesteps in an hour
        self.epTimeStep = config["timestep"]

        # EnergyPlus number of simulation days
        if "days" in config:
            self.simDays = config["days"]
        else:
            self.simDays = 1

        # Number of steps per day
        self.DAYSTEPS = int(24 * self.epTimeStep)

        # Total number of steps
        self.MAXSTEPS = int(self.simDays * self.DAYSTEPS)

        # Time difference between each step in seconds
        self.deltaT = (60 / self.epTimeStep) * 60

        self.outputs = []
        self.inputs = []
        self.kStep = 0
        self.ep = None

        # Z (T + rh + co2), O (T + rh + W)
        observation_space_lb = [0] * 11 + [0] + [0] + [-50, 0] + [0, 0]
        observation_space_ub = [65] * 11 + [1200] + [100] + [60, 100] + [360, 20]

        self.observation_space = spaces.Box(
            np.array(observation_space_lb),
            np.array(observation_space_ub),
            dtype=np.float32,
        )

        self.clg_min = 15
        self.clg_max = 32

        self.rh_min = 1
        self.rh_max = 99

        self.ahu_temp_min = 2
        self.ahu_temp_max = 15

        # Normalized action space
        acts_len = 3
        self.action_space = spaces.Box(np.array([0] * acts_len), np.array([1] * acts_len), dtype=np.float32)


    def construct_inputs(self, action):
        action[0] = action[0] * (self.clg_max - self.clg_min) + (self.clg_min) # cooling
        action[1] = action[1] * (self.rh_max - self.rh_min) + (self.rh_min) # h
        action[2] = action[1] * (self.ahu_temp_max - self.ahu_temp_min) + (self.ahu_temp_min) # ahu

        inp1 = np.clip(action, self.clg_min, self.clg_max)[0]
        inp2 = np.clip(action, self.rh_min, self.rh_max)[1]
        inp3 = np.clip(action, self.ahu_temp_min, self.ahu_temp_max)[2]
        
        print(self.inputs)
        self.inputs = [inp1, inp2, inp3]
        return action
    
    def construct_reward(self, action):
        def temp_penalty(self, cooling_coeff=1):
            res = 0
            for i in range(1,12):
                temp = self.outputs[i]
                if i & 1: # HOT AISLE
                    if temp > 40:
                        res += 500
                    elif temp > 38:
                        res += 200
                else: # COLD AISLE
                    if temp > 30:
                        res += 500
                    elif temp > 27:
                        res += 150
            return res * cooling_coeff
        
        def energy_penalty(self, energy_coeff=0.01): # увеличить коэффициент?
            return self.outputs[0] * energy_coeff
        
        def rh_penalty(self, rh_coeff=1):
            res = 0
            rh =  self.outputs[17]
            if rh > 85:
                res += 100
            if rh < 3:
                res += 25
            return self.outputs[17] * rh_coeff
        
        def action_penalty(self, action, coeff=15000):
            res = 0
            res += max(self.clg_min - action[0], 0)
            res += max(action[0] - self.clg_max, 0)
            if action[0] < 0.05:
                res += 1000

            res += max(self.rh_min - action[1], 0)
            res += max(action[1] - self.rh_max, 0)

            res += max(self.ahu_temp_min - action[2], 0)
            res += max(action[2] - self.ahu_temp_max, 0)

            return res * coeff

        energy_reward = -energy_penalty(self)
        action_reward = -action_penalty(self, action)
        rh_reward = -rh_penalty(self)
        temp_rewatd = -temp_penalty(self)

        reward = (
            energy_reward
            + action_reward
            + rh_reward
            + temp_rewatd
        )
        return reward

    def increment_step_counter(self, time):
        self.kStep += 1
        is_sim_finised = False
        if self.kStep >= (self.MAXSTEPS):
            # last step to close the simulation
            input_packet = encode_packet_simple(self.inputs, time)
            self.ep.write(input_packet)
            # output is empty in the final step
            output_packet = self.ep.read()
            last_output = decode_packet_simple(output_packet)
            is_sim_finised = True
            self.ep.close()
            self.ep = None
        return is_sim_finised
    
    def construct_info(self, time):
        info = {
            'day': int(self.kStep / self.DAYSTEPS) + 1,
            'time': time,
            'cooling_setpoint': self.inputs[0],
            'Humidity_setpoint': self.inputs[1],
            'AHU_Supply_Temp': self.inputs[2],
            
            'Facility_Total_Electricity_Demand_Rate': self.outputs[0],
            'Temp_Z_1': self.outputs[1],
            'Temp_Z_2': self.outputs[2],
            'Temp_Z_3': self.outputs[3],
            'Temp_Z_4': self.outputs[4],
            'Temp_Z_5': self.outputs[5],
            'Temp_Z_6': self.outputs[6],
            'Temp_Z_7': self.outputs[7],
            'Temp_Z_8': self.outputs[8],
            'Temp_Z_9': self.outputs[9],
            'Temp_Z_10': self.outputs[10],
            'Temp_Z_11': self.outputs[11],

            'CO2_Z_1': self.outputs[12],
            'RH_Z_1': self.outputs[13],

            'CO2_Z_5': self.outputs[14],
            'RH_Z_5': self.outputs[15],

            'CO2_Z_6': self.outputs[16],
            'RH_Z_6': self.outputs[17],

            'CO2_Z_7': self.outputs[18],
            'RH_Z_7': self.outputs[19],

            'CO2_Z_11': self.outputs[20],
            'RH_Z_11': self.outputs[21],


            'Outdoor_Air_Drybulb_Temperature': self.outputs[22],
            'Outdoor_Air_Wetbulb_Temperature': self.outputs[23],
            'Outdoor_Air_Relative_Humidity': self.outputs[24],
            'Wind_Speed': self.outputs[25],
            'Wind_Direction': self.outputs[26],

            'Thermal Zone Supply Plenum': self.outputs[27]
            }
        return info
    
    def construct_next_state(self):
        res = [self.outputs[i] for i in range(1, 12)]
        res += [self.outputs[16], self.outputs[17], self.outputs[22], self.outputs[24],
                self.outputs[25], self.outputs[26]]

        return res
    
    def step(self, action):
        time = self.kStep * self.deltaT
        dayTime = time % (60 * 60 * 24)

        if dayTime == 0:
            day_num = int(self.kStep / self.DAYSTEPS) + 1
            print("=" * int(50 * day_num / self.simDays + 1)+ "Day: ", day_num)

        action = self.construct_inputs(action) # also set it in self.input
        input_packet = encode_packet_simple(self.inputs, time)
        self.ep.write(input_packet)
        # after EnergyPlus runs the simulation step, it returns the outputs
        output_packet = self.ep.read()
        self.outputs = decode_packet_simple(output_packet)
        if not self.outputs:
            next_state = self.reset()
            return next_state, 0, False, {}
        reward = self.construct_reward(action)
        next_state = np.array(self.construct_next_state())
        is_sim_finised = self.increment_step_counter(time)
    
        return next_state, reward, is_sim_finised, self.construct_info(time)

    def reset(self):
        if self.ep:
            print("Closing the old simulation and socket.")
            self.ep.close()
            self.ep = None

        print("Starting a new simulation..")
        self.kStep = 0
        idf_dir = os.path.dirname(self.idf_file)
        builder = socket_builder(idf_dir)
        configs = builder.build()
        self.ep = Eplus_contoller.Eplus_controller(
            "localhost", configs[0], self.idf_file, self.weather_file, self.eplus_path
        )

        # read the initial outputs from EnergyPlus
        # these outputs are from warmup phase, so this does not count as a simulation step
        self.outputs = decode_packet_simple(self.ep.read())
        next_state = np.array(self.construct_next_state())
        return next_state
    
