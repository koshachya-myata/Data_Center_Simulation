"""Rise REST server with RL-agent."""
from flask import Flask, jsonify, request, make_response
import numpy as np
import platform
from src.dc_env.data_center_env import DataCenterEplusEnv
import os
import numpy as np
from datetime import datetime
from gymnasium import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3 import PPO
import pandas as pd
import requests
import json


app = Flask(__name__)

pwd = os.getcwd()

TENSORBOARD_LOG = None
MODEL_PATH = os.path.join(pwd, 'models', 'PPO_baseline_230923_233833')

model = PPO.load(MODEL_PATH,
                 custom_objects={
                     'tensorboard_log': TENSORBOARD_LOG
                     })


@app.route('/predict', methods=['POST'])
def obs_post_request():
    data = request.json
    obs = data['observation']
    action, hidden_state = model.predict(observation=obs)

    return jsonify({
        'action': action.tolist()
        })


def raise_server(host='0.0.0.0', port=6113):
    app.run(host=host, port=port)


def get_model_prediction(observation, host='127.0.0.1', port=6113):
    data = {'observation': observation}
    address = 'http://' + host + ':' + str(port) + '/predict'
    response = requests.post(address, json=data)
    if response.status_code == 200:
        result = response.json()
        action = result['action']
        return action
    else:
        print(f'Response code: {response.status_code}')
        return None

if __name__ == "__main__":
    raise_server()
