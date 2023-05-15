import os
import socket

import gymnasium as gym
import numpy as np
from eplus import Eplus_contoller
from eplus.socket_builder import socket_builder
from gymnasium import spaces
from eplus.eplus_packet_contoller import encode_packet_simple,decode_packet_simple

class DataCenterEnv(gym.Env):
    def __init__(self, config):
        cur_dir = os.path.dirname(__file__)
        self.idf_file = cur_dir + "/buildings/1ZoneDataCenterCRAC_wPumpedDXCoolingCoil/1ZoneDataCenterCRAC_wPumpedDXCoolingCoil.idf"

        self.weather_file = cur_dir + "/" + config["weather_file"]

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

        # state can be all the inputs required to make a control decision
        # getting all the outputs coming from EnergyPlus for the time being
        self.observation_space = spaces.Box(
            np.array([0, -50, 0, 0, 0]),  # zone temp, outdoor drybulb temp, relative humidity, zone Humidity, wind speed
            np.array([60, 70, 100, 100, 20]),
            dtype=np.float32,
        )

        self.clg_min = 16
        self.clg_max = 32

        self.htg_min = 5
        self.htg_max = 20 

        self.min_vent_lb = 10
        self.min_vent_ub = 30

        self.max_vent_lb = 20
        self.max_vent_ub = 40

        # Normalized action space
        acts_len = 6
        self.action_space = spaces.Box(np.array([0] * acts_len), np.array([1] * acts_len), dtype=np.float32)


    def construct_inputs(self, action):
        # actioins is normalized
        action[0] = action[0] * (self.clg_max - self.clg_min) + (self.clg_min) # cooling
        action[1] = action[1] * (self.htg_max - self.htg_min) + (self.htg_min) # heating
        action[3] = action[3] * (self.min_vent_ub - self.min_vent_lb) + self.min_vent_lb # min vent
        action[4] = action[4] * (self.max_vent_ub - self.max_vent_lb) + self.max_vent_lb # max vent
        action[5] = action[5] * (50 - (-50)) + (-50) # vent delta
        # force action to be within limits
        cooling_setpoint = np.clip(action, self.clg_min, self.clg_max)[0]
        heating_setpoint = np.clip(action, self.htg_min, self.htg_max)[1]
        flow_rate_f_coeff = action[2]
        vent_min_indoor_temp = np.clip(action, self.min_vent_lb, self.min_vent_ub)[3]
        vent_max_indoor_temp = np.clip(action, self.max_vent_lb, self.max_vent_ub)[4]
        vent_delta = np.clip(action, -50, 50)[5]

        self.inputs = [cooling_setpoint, heating_setpoint, flow_rate_f_coeff,
                    vent_min_indoor_temp, vent_max_indoor_temp, vent_delta]
        
        return action
    
    def construct_reward(self, action):  # ! FIX ME
        # reward needs to be a combination of energy and comfort requirement
        energy_coeff = -0.000005
        heating_coeff = -100
        cooling_coeff = -100
        energy = self.outputs[0]
        zone_temperature = self.outputs[1]  # taking mid-zone 2 as an example
        heating_setpoint = 18  # fixed lower limit in celcius
        cooling_setpoint = 27  # fixed upper limit in celcius
        heating_penalty = max(heating_setpoint - zone_temperature, 0)
        cooling_penalty = max(zone_temperature - cooling_setpoint, 0)

        # punish if action is out of limits
        action_penalty_coeff = -75
        max_penalty = max(self.clg_min - action[0], 0)
        min_penalty = max(action[0] - self.clg_max, 0)
        action_penalty = action_penalty_coeff * (max_penalty + min_penalty)
        max_penalty = max(self.htg_min - action[1], 0)
        min_penalty = max(action[1] - self.htg_max, 0)
        action_penalty += action_penalty_coeff * (max_penalty + min_penalty)

        # final reward
        reward = (
            energy_coeff * energy
            + heating_coeff * heating_penalty
            + cooling_coeff * cooling_penalty
            + action_penalty
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
    
    def return_info(self, time):
        info = {
            'day': int(self.kStep / self.DAYSTEPS) + 1,
            'time': time,
            'cooling_setpoint': self.inputs[0],
            'heating_setpoint': self.inputs[1],
            'flow_rate_f_coeff': self.inputs[2],
            'vent_min_indoor_temp': self.inputs[3],
            'vent_max_indoor_temp': self.inputs[4],
            'vent_delta': self.inputs[5],
            'Facility_Total_Electricity_Demand_Rate': self.outputs[0],
            'Zone_Air_Temperature': self.outputs[1],
            'Site_Outdoor_Air_Drybulb_Temperature': self.outputs[2],
            'Outdoor_Air_Wetbulb_Temperature': self.outputs[3],
            'Outdoor_Air_Relative_Humidity': self.outputs[4],
            'Wind_Speed': self.outputs[5],
            'Wind_Direction': self.outputs[6],
            'CO2_ppm': self.outputs[7],
            'Zone_Air_Relative_Humidity' : self.outputs[8]
            }
        return info

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
        next_state = np.array([self.outputs[1], self.outputs[2], self.outputs[4], 
                            self.outputs[-1], self.outputs[5]])
        is_sim_finised = self.increment_step_counter(time)
    
        return next_state, reward, is_sim_finised, self.return_info(time)

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
        #print('OUTPUT:-----------------------', self.outputs)
        next_state = np.array([self.outputs[1], self.outputs[2], self.outputs[4], 
                            self.outputs[-1], self.outputs[5]])
        return next_state
    
