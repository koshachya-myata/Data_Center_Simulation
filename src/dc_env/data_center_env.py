import os

import gymnasium as gym
import numpy as np
from ..eplus_controll import Eplus_contoller
from ..eplus_controll.socket_builder import socket_builder
from gymnasium import spaces
from ..eplus_controll.eplus_packet_contoller import (
                        encode_packet_simple, decode_packet_simple)
import platform


class DataCenterEnv(gym.Env):
    def __init__(self, config):
        super(DataCenterEnv, self).__init__()
        os_type = platform.system()
        pwd_del = '/'
        if os_type == "nt":  # windows
            pwd_del = '\\'

        cur_dir = os.path.dirname(__file__)
        self.idf_file = cur_dir + pwd_del + "buildings" + pwd_del + \
            "MultiZone_DC_wHot_nCold_Aisles" + pwd_del + "MultiZone_DC.idf"

        self.weather_file = cur_dir + pwd_del + config["weather_file"]

        if "eplus_path" in config:
            self.eplus_path = config["eplus_path"]
        else:
            self.eplus_path = "/Applications/EnergyPlus-23-1-0/"

        # EnergyPlus number of timesteps in an hour
        self.epTimeStep = config["timestep"]

        if "verbose" in config:
            self.verbose = config['verbose']
        else:
            self.verbose = 1

        # EnergyPlus number of simulation days
        if "days" in config:
            self.simDays = config["days"]
        else:
            self.simDays = 1

        # Number of steps per day
        self.DAYSTEPS = int(24 * self.epTimeStep)
        print('simDays', self.simDays)
        # Total number of steps
        self.MAXSTEPS = int(self.simDays * self.DAYSTEPS)
        print('MAXSTEPS', self.MAXSTEPS
              )
        # Time difference between each step in seconds
        self.deltaT = (60 / self.epTimeStep) * 60

        self.outputs = []
        self.inputs = []
        self.kStep = 0
        self.ep = None
        # self.clip_obs = True

        # Z (T + co2 + rh), O (T + rh + W (speed + dir))
        # observation_space_lb = [0] * 11 + [0] + [0] + [-50, 0] + [0, 0]
        # observation_space_ub = [50] * 11 + [2500] + [100] +
        #                       [60, 100] + [25, 360]

        observation_space_lb = [0] * 3 + [0] + [-50]  # 3T + rh + O(t)
        observation_space_ub = [50] * 3 + [100] + [60]

        self.observation_space = spaces.Box(
            np.array(observation_space_lb),
            np.array(observation_space_ub),
            dtype=np.float32,
        )

        self.clg_min = 15
        self.clg_max = 32

        self.rh_min = 1
        self.rh_max = 99

        self.ahu_temp_min = 4
        self.ahu_temp_max = 16

        # Normalized action space
        acts_len = 3
        self.action_space = spaces.Box(np.array([0] * acts_len),
                                       np.array([1] * acts_len),
                                       dtype=np.float32)

    def construct_inputs(self, action):  # action is readonly
        """
        USE IF INPUTS IS NORMALIZED
        """
        # convert $x \in  [0, 1]$ to $x \in
        # [action_min_value, action_max_value]$
        inp1 = action[0] * (self.clg_max - self.clg_min) + \
            (self.clg_min)  # cooling
        inp2 = action[1] * (self.rh_max - self.rh_min) + \
            (self.rh_min)  # rh
        inp3 = action[2] * (self.ahu_temp_max - self.ahu_temp_min) + \
            (self.ahu_temp_min)  # ahu
        action_values = [inp1, inp2, inp3]

        # clip actions to range
        inp1_ranged = np.clip(action_values, self.clg_min, self.clg_max)[0]
        inp2_ranged = np.clip(action_values, self.rh_min, self.rh_max)[1]
        inp3_ranged = np.clip(action_values, self.ahu_temp_min,
                              self.ahu_temp_max)[2]

        if self.verbose == 2:
            print(action_values)
        self.inputs = [inp1_ranged, inp2_ranged, inp3_ranged]
        return action_values

    def construct_reward(self, action):

        def temp_penalty(self, hot_aisle_coeff=0.5, cold_aisle_coeff=1,
                         reward_if_not_cold=False):
            res = 0
            for i in range(1, 12):
                temp = self.outputs[i]
                if i & 1:  # HOT AISLE
                    if temp > 44:
                        res += (temp - 44) * hot_aisle_coeff
                    if temp > 50:
                        res += (temp - 44) * hot_aisle_coeff * 2
                else:  # COLD AISLE
                    if temp > 25:
                        res += (temp - 25) * cold_aisle_coeff
                    if temp > 35:
                        res += (temp - 25) * cold_aisle_coeff * 3
                    if reward_if_not_cold and temp >= 16.5 and temp < 25:
                        res -= 0.5
            return res

        def energy_penalty(self, energy_coeff=0.00001):
            return self.outputs[0] * energy_coeff

        def rh_penalty(self, lb_coeff=0.5, ub_coeff=1):
            res = 0
            rh = self.outputs[17]
            if rh > 90:
                res += (rh - 90) * ub_coeff
            if rh < 3:
                res += (3 - rh) * lb_coeff
            return res

        def action_penalty(self, action, coeff=5, force_action=False):
            res = 0
            res += max(self.clg_min - action[0] + 0.1, 0)
            res += max(action[0] - self.clg_max + 0.1, 0)

            # res += max(self.rh_min - action[1] + 0.1, 0)
            # res += max(action[1] - self.rh_max + 0.1, 0)

            res += max(self.ahu_temp_min - action[2] + 0.1, 0)
            res += max(action[2] - self.ahu_temp_max + 0.1, 0)

            if force_action:
                if action[0] <= 16:
                    res += (16 - action[0]) * 2
                if action[1] <= 3:
                    res += (3 - action[1]) * 0.5
                if action[1] > 95:
                    res += (action[1] - 95) * 0.5
                if action[2] < 5:
                    res += (5 - action[2]) * 1.5
                if action[2] > 16:
                    res += (action[2] - 16) * 2

            return res * coeff

        # print('Getted action:', action)
        energy_reward = -energy_penalty(self, energy_coeff=0.00002)
        action_reward = -action_penalty(self, action, force_action=True)
        rh_reward = -rh_penalty(self)
        temp_reward = -temp_penalty(self, reward_if_not_cold=True)
        # print('Energy penalty', energy_reward)
        # print('Action penalty', action_reward)
        # print('RH penalty', rh_reward)
        # print('Temperature penalty', temp_reward)
        reward = (
            energy_reward
            + action_reward
            + rh_reward
            + temp_reward
        )
        # print('REWARD:', reward)
        return reward if reward else 0

    def increment_step_counter(self, time):
        self.kStep += 1

        is_sim_finised = False
        if self.kStep >= (self.MAXSTEPS):
            if self.verbose >= 1:
                print('LAST STEP')
            # last step for closing the simulation
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
        # res = [self.outputs[i] for i in range(1, 12)]
        # res += [self.outputs[16], self.outputs[17], self.outputs[22],
        #   self.outputs[24],
        #         self.outputs[25], self.outputs[26]]
        # print('NEW STATE', res)
        res = [self.outputs[i] for i in [2, 6, 10]]
        res += [self.outputs[17], self.outputs[22]]

        return res

    def step(self, action):
        # print('ACTION:', action)
        time = self.kStep * self.deltaT
        dayTime = time % (60 * 60 * 24)

        if dayTime == 0:
            day_num = int(self.kStep / self.DAYSTEPS) + 1
            if self.verbose >= 1:
                print("=" * int(50 * day_num / self.simDays + 1) +
                      "Day: ", day_num)

#        if action[0] != 1 and action[0] != 0:
#            print(action)
        action = self.construct_inputs(action)  # also set it in self.input
        if self.verbose >= 3:
            print('NEW ACTION', action)
        input_packet = encode_packet_simple(self.inputs, time)
        self.ep.write(input_packet)
        # after EnergyPlus runs the simulation step, it returns the outputs
        output_packet = self.ep.read()
        self.outputs = decode_packet_simple(output_packet)
        if not self.outputs:
            print('NO OUTPUT FROM ENERGYPLUS')
            next_state = self.reset()
            return next_state, 0, True, False, {}
        reward = self.construct_reward(action)
        next_state = np.array(self.construct_next_state())
        is_sim_finised = self.increment_step_counter(time)
        truncated = False
        if self.verbose >= 4:
            print('END STEP, STATE:', next_state)
            print('END STEP. REWARD: ', reward)
            print('END STEP. is_sim_finised: ', is_sim_finised)
            print('END STEP INFO: ', self.construct_info(time))
        return next_state, reward, is_sim_finised, truncated,\
            self.construct_info(time)

    def reset(self, seed=None, options=None):
        if self.verbose >= 4:
            print('RESET' * 10)
        super().reset(seed=seed)
        if self.ep:
            if self.verbose >= 1:
                print("Closing the old simulation and socket.")
            self.ep.close()
            self.ep = None

        if self.verbose >= 1:
            print("Starting a new simulation..")
        self.kStep = 0
        idf_dir = os.path.dirname(self.idf_file)
        builder = socket_builder(idf_dir)
        configs = builder.build()
        self.ep = Eplus_contoller.Eplus_controller(
            "localhost",
            configs[0],
            self.idf_file,
            self.weather_file,
            self.eplus_path
        )

        # read the initial outputs from EnergyPlus
        # these outputs are from warmup phase, so
        # this does not count as a simulation step
        self.outputs = decode_packet_simple(self.ep.read())
        next_state = np.array(self.construct_next_state())
        if self.verbose >= 4:
            print('First outputs: ', next_state)
        return next_state, {}
