"""Rise http-server with RL-agent."""
from typing import Union
from flask import Flask, jsonify, request
import os
from stable_baselines3 import PPO
import requests


app = Flask(__name__)

pwd = os.getcwd()

TENSORBOARD_LOG = None
MODEL_PATH = os.path.join(pwd, 'models', 'new_PPO_baseline_231019_133911')

model = PPO.load(MODEL_PATH,
                 custom_objects={
                     'tensorboard_log': TENSORBOARD_LOG
                     })


@app.route('/predict', methods=['POST'])
def obs_post_request():
    """
    Get observation from post method and return action to client side.

    Returns:
        Response: json with action list.
    """
    data = request.json
    obs = data['observation']
    action, hidden_state = model.predict(observation=obs)

    return jsonify({
        'action': action.tolist()
        })


def raise_server(host: str = '0.0.0.0', port: int = 6113):
    """
    Raise server on host:port.

    Args:
        host (str, optional): host. Defaults to '0.0.0.0'.
        port (int, optional): port. Defaults to 6113.
    """
    app.run(host=host, port=port)


def get_model_prediction(observation: list, host: str = '127.0.0.1',
                         port: int = 6113) -> Union[list[float], None]:
    """
    Get model prediction from http host:port based on observation.

    Args:
        observation (list): observations
        host (str, optional): host. Defaults to '127.0.0.1'.
        port (int, optional): port. Defaults to 6113.

    Returns:
        Union[list[float], None]: list of actions or None.
    """
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
