import platform
import os
import numpy as np
from datetime import datetime
from gymnasium import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3 import PPO
import pandas as pd
from src.dc_env.data_center_env import DataCenterEplusEnv
from src.inference.model_api import get_model_prediction
from src.inference.DBApi import DBApi
from src.dc_env.make_config import make_config


def init_db(table_name='DC_log', drop_if_exists=False):
    db = DBApi(host='localhost',
               port='32023',
               database='datacenter',
               login='admin',
               password='admin')
    db.create_db(database_name='datacenter')
    if drop_if_exists:
        drop_table_sql = f"DROP TABLE IF EXISTS {db.database}.{table_name}"
        db.run_sql(drop_table_sql)
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {db.database}.{table_name}
    (
    cooling_setpoint Float32,
    humidity_setpoint Float32,
    ahu_supply_temp Float32,
    facility_total_electricity_demand_rate Float32,
    temp_z_1 Float32,
    temp_z_2 Float32,
    temp_z_3 Float32,
    temp_z_4 Float32,
    temp_z_5 Float32,
    temp_z_6 Float32,
    temp_z_7 Float32,
    temp_z_8 Float32,
    temp_z_9 Float32,
    temp_z_10 Float32,
    temp_z_11 Float32,
    co2_z_1 Float32,
    rh_z_1 Float32,
    co2_z_5 Float32,
    rh_z_5 Float32,
    co2_z_6 Float32,
    rh_z_6 Float32,
    co2_z_7 Float32,
    rh_z_7 Float32,
    co2_z_11 Float32,
    rh_z_11 Float32,
    outdoor_air_drybulb_temperature Float32,
    outdoor_air_wetbulb_temperature Float32,
    outdoor_air_relative_humidity Float32,
    wind_speed Float32,
    wind_direction Float32,
    thermal_zone_supply_plenum Float32
    )
    ENGINE = MergeTree
    ORDER BY tuple()
    """
    _ = db.run_sql(create_table_sql)
    return db

def insert_to_table(db, info, table_name='DC_log'):
    info = {key: round(value, 6) for key, value in info.items()}
    insert_sql = f"""
    INSERT INTO {db.database}.{table_name} (*) VALUES (
            {info['cooling_setpoint']},
            {info['Humidity_setpoint']},
            {info['AHU_Supply_Temp']},
            {info['Facility_Total_Electricity_Demand_Rate']},
            {info['Temp_Z_1']},
            {info['Temp_Z_2']},
            {info['Temp_Z_3']},
            {info['Temp_Z_4']},
            {info['Temp_Z_5']},
            {info['Temp_Z_6']},
            {info['Temp_Z_7']},
            {info['Temp_Z_8']},
            {info['Temp_Z_9']},
            {info['Temp_Z_10']},
            {info['Temp_Z_11']},
            {info['CO2_Z_1']},
            {info['RH_Z_1']},
            {info['CO2_Z_5']},
            {info['RH_Z_5']},
            {info['CO2_Z_6']},
            {info['RH_Z_6']},
            {info['CO2_Z_7']},
            {info['RH_Z_7']},
            {info['CO2_Z_11']},
            {info['RH_Z_11']},
            {info['Outdoor_Air_Drybulb_Temperature']},
            {info['Outdoor_Air_Wetbulb_Temperature']},
            {info['Outdoor_Air_Relative_Humidity']},
            {info['Wind_Speed']},
            {info['Wind_Direction']},
            {info['Thermal_Zone_Supply_Plenum']}
        )
    """
    res = db.run_sql(insert_sql)
    return res


def simulate_agent_inference(init_obs, sim_days=8):
    env_config, horizon, pwd = make_config(sim_days=sim_days)
    db = init_db(drop_if_exists=True)
    env = DataCenterEplusEnv(env_config)  # change env for real inference
    env.reset()
    action = get_model_prediction(observation=init_obs)
    is_sim_finised = is_turncated = False
    while not is_sim_finised and not is_turncated:
        obs, reward, is_sim_finised, is_turncated, info = env.step(action)
        insert_to_table(db, info, 'DC_log')
        action = get_model_prediction(observation=obs.tolist())
    print('Sumulation Finished')
    db.close_connection()
