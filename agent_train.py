import platform
import random
import pandas as pd
from eplus import DataCenterEnv
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
#import keras.api._v2.keras as keras

from rl.agents import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from tensorflow import keras

os_type = platform.system()
if os_type == "Linux":
    eplus_path = "/usr/local/EnergyPlus-8-8-0/"
elif os_type == "nt":  # windows
    eplus_path = "C:\\EnergyPlus-8-8-0\\"
else:  # mac
    eplus_path = "/Applications/EnergyPlus-23-1-0/"
config = {
    "eplus_path": eplus_path,
    "weather_file": 'weather/Moscow.epw', #"weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
    'days': 142,
    'timestep': 12
}
pwd = os.getcwd()

l_states = 5
l_act = 6

model = Sequential()
model.add(Flatten(input_shape=(1, l_states)))
model.add(Input(shape=(l_states, 1)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(l_act, activation='linear'))


print('!!!!!!!!!!!', model.predict(np.array([1,2,3,4,5]).reshape(1, 1, -1)))
agent = DDPGAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=l_act,
    nb_steps_warmup=2,
    target_model_update=0.01
)

env = DataCenterEnv(config)

print(env)
print(env.observation_space)

agent.compile(Adam(lr=0.01), metrics=["mae"])
agent.fit(env, nb_steps=10, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=False)
print(np.mean(results.history["episode_reward"]))

agent.save(pwd + '/DQNAgent.h5')

env.close()

if __name__ == "__main__":
    print('FIX ME')
    exit(0)