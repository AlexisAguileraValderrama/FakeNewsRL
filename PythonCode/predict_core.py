from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


import os
from FakeNewsEnv import FakeNewsEnv

from datetime import datetime

model_name = 'DQN'
brain_name = 'DQN brain - 03 02 2023, 07:16:14 - 00100'
brain_checkpoint = 'checkpoint 400000 - 500000'
brain_replay = 'replay 400000 - 500000'

eval_outcome_logs = f"./models/{model_name}/{brain_name}/outcome_logs/eval/"

model_info = {'model_name': model_name,
              'brain_name' : 'DQN brain - 03 02 2023, 07:16:14 - 00100',
            'brain_checkpoint' : 'checkpoint 400000 - 500000',
            'flags' : brain_name[-5:],
            'brain_replay' : 'replay 400000 - 500000',
            'eval_outcome_logs' : eval_outcome_logs}

if not os.path.exists(model_info['eval_outcome_logs'] + '/bin/'):
    os.makedirs(model_info['eval_outcome_logs'] + 'bin/')

if not os.path.exists(model_info['eval_outcome_logs'] + '/plain/'):
    os.makedirs(model_info['eval_outcome_logs'] + 'plain/')

flags = model_info['flags']
vec_env = FakeNewsEnv(flags,train_mode = True,model_name = model_info['model_name'],local_data_path='/home/serapf/Desktop/FakeNewsRL/chunk_600-900')

brain_path = f"./models/{model_info['model_name']}/{model_info['brain_name']}/{model_info['brain_checkpoint']}.zip"
model = DQN.load(brain_path,vec_env)

replay_path = f"./models/{model_info['model_name']}/{model_info['brain_name']}/{model_info['brain_replay']}.pkl"
model.load_replay_buffer(replay_path)

obs = vec_env.reset()
counter = 0

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    
    if dones:
        counter = counter + 1
        vec_env.reset()

    if counter == 300:
        break

#fecha de la creacion del modelo
now = datetime.now() # current date and time
date_time = now.strftime("%m %d %Y, %H:%M:%S")

vec_env.saveOutcomelog(path_name_binary=model_info['eval_outcome_logs']+'/bin/'+date_time,
                       path_name_plain=model_info['eval_outcome_logs']+'/plain/'+date_time)