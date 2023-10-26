"""Data center environment classes."""

from typing import Dict, Any, TypeVar, SupportsFloat, Union
from abc import ABC, abstractmethod
import os

import gymnasium as gym
import numpy as np
from ..eplus_controll import EplusController
from ..eplus_controll.SocketBuilder import SocketBuilder
from gymnasium import spaces
from ..eplus_controll.eplus_packet_contoller import (
                        encode_packet_simple, decode_packet_simple)


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def denormalize(num: float, a_min: float, a_max: float, norm_type=2) -> float:
    """
    Denormalze num, using a_min and a_max.

    If norm_type is 1: denormalize num from [0, 1].
    If norm_type is 2: denormalize num from [-1, 1].

    Args:
        num (float): Number to denormalize
        a_min (float): Action lower border.
        a_max (float): Action upper border.
        norm_type (int, optional): Normalize type. Defaults to 2.

    Returns:
        float: denormalized number.
    """
    res = num
    if norm_type == 1:
        res = num * (a_max - a_min) + a_min
    if norm_type == 2:
        res = (num + 1) / 2 * (a_max - a_min) + a_min
    return res


def normalize(num: float, a_min: float, a_max: float, norm_type=2) -> float:
    """
    Normalize num using a_min and and a_max.

    If norm_type is 1: normalize num to [0, 1].
    If norm_type is 2: normalize num to [-1, 1].

    Args:
        num (float): Number to normalize.
        a_min (float): Action lower border.
        a_max (float): Action upper border.
        norm_type (int, optional): Normalize type. Defaults to 2.

    Returns:
        float: normalized number.
    """
    res = num
    if norm_type == 1:
        res = (num - a_min) / (a_max - a_min)
    if norm_type == 2:
        res = (num - a_min) / (a_max - a_min) * 2 - 1
    return res


def convert_joules_to_watts(joules: float, timestamp_per_hour: int) -> float:
    """
    Convert Joules to Watts for process with timestamp_per_hour frequency.

    Args:
        joules (float): Joules.
        timestamp_per_hour (int): process per hour frequency.

    Returns:
        float: Watts.
    """
    return joules / (60 / timestamp_per_hour * 60)


class DataCenterEnv(ABC, gym.Env):
    """Defines common logic for data center environment classes."""

    def __init__(self):
        """Initiate the gym environment and basic class properties."""
        super(DataCenterEnv, self).__init__()
        # 3T + rh + O(t) + prev[Col, Ahu]
        observation_space_lb = [0] * 3 + [0] + [0] + [-1] * 2
        observation_space_ub = [1] * 3 + [1] + [1] + [1] * 2

        self.outputs = []
        self.inputs = []

        self.obs_boundaries = [
            [0, 50],
            [0, 50],
            [0, 50],
            [0, 100],
            [-50, 60],
        ]

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

        self.timestamps_in_hour = 5

        # Normalized action space
        acts_len = 3
        self.action_space = spaces.Box(np.array([-1] * acts_len),
                                       np.array([1] * acts_len),
                                       dtype=np.float32)

        self.prev_actions = None

    def construct_info(self) -> Dict[str, Any]:
        """
        Construct basic log info.

        Returns:
            Dict[str, float]: dictionary with info.
        """
        info = {
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

            'Thermal_Zone_Supply_Plenum': self.outputs[27],
            'Air_System_Total_Cooling_Energy': convert_joules_to_watts(
                                            self.outputs[28],
                                            self.timestamps_in_hour
                                        )
            }
        return info

    def construct_inputs(self, action: list[float],
                         norm_type=2) -> list[float]:
        """
        Translate input into the correct format for actions.

        Set in self.inputs clipped actions.
        Return without clipping.
        norm_type:
            if 0 => dont denormalize
            if 1 => from [0, 1]
            if 2 => from [-1, 1] (prefer)

        Args:
            norm_type (int, optional): Normalize type. Defaults to 2.

        Returns:
            list[float]: correct actions without clipping.
        """
        # convert $x \in  [0, 1]$ to $x \in
        # [action_min_value, action_max_value]$
        inp1 = denormalize(action[0], self.clg_min, self.clg_max, norm_type)
        inp2 = denormalize(action[1], self.rh_min, self.rh_max, norm_type)
        inp3 = denormalize(action[2], self.ahu_temp_min, self.ahu_temp_max,
                           norm_type)
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

    def construct_reward(self, action: list[float]) -> float:
        """
        Construct reward based on actions, input, DC info.

        Args:
            action (list[float]): actions.
        Returns:
            float: reward
        """
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
                    if temp > 24.5:
                        res += (temp - 24.5) * cold_aisle_coeff
                    if temp > 35:
                        res += (temp - 24.5) * cold_aisle_coeff * 3
                    if reward_if_not_cold and temp >= 16.5 and temp < 24.5:
                        res -= 0.5
            return res

        def energy_penalty(self, energy_coeff=0.00001):
            return convert_joules_to_watts(
                        self.outputs[28],
                        self.timestamps_in_hour
                    ) * energy_coeff

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

        energy_reward = -energy_penalty(self, energy_coeff=0.00002)
        action_reward = -action_penalty(self, action, force_action=True)
        rh_reward = -rh_penalty(self)
        temp_reward = -temp_penalty(self, reward_if_not_cold=False)

        reward = (
            energy_reward
            + action_reward
            + rh_reward
            + temp_reward
        )
        # print('REWARD:', reward)
        return reward if reward else 0

    def construct_next_state(self) -> list[float]:
        """
        Construct data needed for next step.

        Returns:
            list[float]: list with data.
        """
        res = [self.outputs[i] for i in [2, 6, 10]]
        res += [self.outputs[17], self.outputs[22]]
        for i in range(len(res)):
            res[i] = normalize(res[i], self.obs_boundaries[i][0],
                               self.obs_boundaries[i][1], norm_type=1)
        if self.prev_actions is not None:
            res += [self.prev_actions[0]]
            res += [self.prev_actions[2]]
        else:
            res += [0.1, 0.12]  # default setpoints

        return res

    @abstractmethod
    def realize_action(self, action: list[float]) -> tuple[ObsType,
                                                           SupportsFloat,
                                                           bool,
                                                           bool,
                                                           dict[str, Any]]:
        """
        Realze action.

        Args:
            action (list[float]): actions list.
        Returns:
            tuple[ObsType, SupportsFloat, bool
                  bool, dict[str, Any]]: next state, reward, flag: sim_finised,
                                         flag: truncated, info dict.
        """
        pass

    def step(self, action: list[float]) -> tuple[ObsType,
                                                 SupportsFloat,
                                                 bool,
                                                 bool,
                                                 dict[str, Any]]:
        """
        Environment step.

        Args:
            action (list[float]): list with actions.

        Returns:
            tuple[ObsType, SupportsFloat, bool
                  bool, dict[str, Any]]: next state, reward, flag: sim_finised,
                                         flag: truncated, info dict.
        """
        self.prev_actions = action.copy()
        action = self.construct_inputs(action)
        next_state, reward, is_sim_finised, truncated, info = \
            self.realize_action(action)
        reward = self.construct_reward(action)
        return next_state, reward, is_sim_finised, truncated, info

    @abstractmethod
    def reset(self, seed: Union[int, None] = None,
              options: dict[str, Any] = None) -> tuple[ObsType,
                                                       dict[str, Any]]:
        """
        Reset environment.

        Args:
            seed (Union[int, None], optional): random seed. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        """
        pass


class DataCenterEplusEnv(DataCenterEnv):
    """DC env for simulation. Use EnergyPlus."""

    def __init__(self, config: dict[str, Any]):
        """
        Initiate the environment and basic class properties based on config.

        Args:
            config (config: dict[str, Any]): _description_
        """
        super().__init__()

        cur_dir = os.path.dirname(__file__)
        self.idf_file = os.path.join(cur_dir, 'buildings',
                                     'MultiZone_DC_wHot_nCold_Aisles',
                                     'MultiZone_DC.idf')

        self.weather_file = os.path.join(cur_dir, config["weather_file"])
        if "eplus_path" in config:
            self.eplus_path = config["eplus_path"]
        else:
            self.eplus_path = "/Applications/EnergyPlus-23-1-0/energyplus"

        # EnergyPlus number of timesteps in an hour
        self.epTimeStep = config["timestep"]
        self.timestamps_in_hour = self.epTimeStep

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

        self.kStep = 0
        self.ep = None

    def increment_step_counter(self, time: int) -> bool:
        """
        Increment self.kStep. If it reaches MAXSTEPS, close EP process.

        Args:
            time (int): _description_

        Returns:
            bool: Is EP simulation finished?
        """
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
            _ = decode_packet_simple(output_packet)  # last_output
            is_sim_finised = True
            self.ep.close()
            self.ep = None
        return is_sim_finised

    def construct_info(self, time: int) -> Dict[str, Union[int, float]]:
        """
        Construct DC log info.

        Returns:
            Dict[str, Union[int, float]]: dictionary with info.
        """
        info = super().construct_info()
        info['day'] = int(self.kStep / self.DAYSTEPS) + 1
        info['time'] = time
        return info

    def realize_action(self, action: list[float]) -> tuple[ObsType,
                                                           SupportsFloat,
                                                           bool,
                                                           bool,
                                                           dict[str, Any]]:
        """
        Send actions to E+, get a response, return data for the next step.

        Args:
            action (list[float]): actions list.
        Returns:
            tuple[ObsType, SupportsFloat, bool
                  bool, dict[str, Any]]: next state, reward, flag: sim_finised,
                                         flag: truncated, info dict.
        """
        time = self.kStep * self.deltaT
        day_time = time % (60 * 60 * 24)
        if day_time == 0:
            day_num = int(self.kStep / self.DAYSTEPS) + 1
            if self.verbose >= 1:
                print("=" * int(50 * day_num / self.simDays + 1) +
                      "Day: ", day_num)

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
        return next_state, reward, is_sim_finised, truncated, \
            self.construct_info(time)

    def reset(self, seed: Union[int, None] = None,
              options: dict[str, Any] = None) -> tuple[ObsType,
                                                       dict[str, Any]]:
        """
        Reset env. Stop EP process if exist.

        Args:
            seed (int | None, optional): random seed. Defaults to None.
            options (dict[str, Any], optional): reset options.
                                                Defaults to None.

        Returns:
             tuple[ObsType, dict[str, Any]]: next state, info.
        """
        if self.verbose >= 4:
            print('RESET' * 10)
        super().reset(seed=seed)
        if self.ep:
            if self.verbose >= 1:
                print('Closing the old simulation and socket.')
            self.ep.close()
            self.ep = None

        if self.verbose >= 1:
            print('Starting a new simulation.')
        self.kStep = 0
        idf_dir = os.path.dirname(self.idf_file)
        builder = SocketBuilder(idf_dir)
        configs = builder.build()
        self.ep = EplusController(
            'localhost',
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
