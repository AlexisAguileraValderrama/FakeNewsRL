#Esta clase funciona para definir los parametros y procesos
#Para entrenar el modelo 

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C


import os
from FakeNewsEnv import FakeNewsEnv

from datetime import datetime

import pickle
prop_flags = ['00100','10100','01100','00110','00101','11100','10110','10101','11110','01111','10111','11111']
models_name = ['DQN','A2C','PPO']

# for mod in models_name:
#     for fla in prop_flags:

#fecha de la creacion del modelo
now = datetime.now() # current date and time
date_time = now.strftime("%m %d %Y, %H:%M:%S")

flags = '00100'
model_name = 'DQN'
brain_name = f'{model_name} brain - {date_time} - {flags}'

checkpoints_path = f"./models/{model_name}/{brain_name}/"
outcome_logs = f"./models/{model_name}/{brain_name}/outcome_logs/"

model_info = {'name': model_name,
            'brain_name' : brain_name,
            'checkpoints_path' : checkpoints_path,
            'outcome_logs' : outcome_logs,
            'path_load':None}

##Carpetas para hacer guardar logs para tensorboard y modelos
# model_path = 
# checkpoints_path = f"models/{model_name}"
logdir = "logs"

if not os.path.exists(model_info['checkpoints_path']):
    os.makedirs(model_info['checkpoints_path'])

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(model_info['outcome_logs'] + 'bin/'):
    os.makedirs(model_info['outcome_logs'] + 'bin/')

if not os.path.exists(model_info['outcome_logs'] + 'plain/'):
    os.makedirs(model_info['outcome_logs'] + 'plain/')


#Creacion del ambiente
# 0: Similitud al insertar
# 1: Longitud de sent insertado
# 2: Descision diff
# 3: Reward por repetir step
# 4: Reward por el tamaño de cada lista

env = FakeNewsEnv(flags,train_mode = True,model_name = model_name,local_data_path='/home/serapf/Desktop/FakeNewsRL/train_chunk_0-1164')

#Creación de modelo de RL PPO
if model_info['name'] == 'PPO':
    try:
        model = PPO.load(model_info['path_load'],env)
        print(f"Modelo {model_info['path_load']} PPO se cargo")
    except:
        model = PPO("MlpPolicy",env,verbose = 1, tensorboard_log=logdir)
        print(f'Modelo PPO nuevo Creado')

elif model_info['name'] == 'DQN':
    try:
        model = DQN.load(model_info['path_load'],env)
        print(f"Modelo {model_info['path_load']} DQN se cargo")
    except:
        model = DQN("MlpPolicy",env,verbose = 1, tensorboard_log=logdir, buffer_size=50000)
        print(f'Modelo DQN nuevo Creado')

elif model_info['name'] == 'A2C':
    try:
        model = A2C.load(model_info['path_load'],env)
        print(f"Modelo {model_info['path_load']} A2C se cargo")
    except:
        model = A2C("MlpPolicy",env,verbose = 1, tensorboard_log=logdir)
        print(f'Modelo A2C nuevo Creado')


#Total de pasos por hacer
TIMESTEPS = 500000

CHECKPOINTS = 5

TIMESTEPS_PER_CHECKPOINT = TIMESTEPS // CHECKPOINTS

EPISODIES_PER_LOG = 5


#Loop para ir guardando los modelos

input('Presiona enter para iniciar entrenamiento')

#COmenzar el entrenamiento del modelo 
log_name = f'{model_name} - {date_time} - {flags}'
for i in range(CHECKPOINTS):
    checkpoint_name = f"{i*TIMESTEPS_PER_CHECKPOINT} - {(i+1)*TIMESTEPS_PER_CHECKPOINT}"
    model.learn(total_timesteps=TIMESTEPS_PER_CHECKPOINT, reset_num_timesteps=False, log_interval=EPISODIES_PER_LOG, tb_log_name=log_name)
    #Guardar el checkpoint del modelo
    print("saving the model...")
    save_path = model_info['checkpoints_path']+"checkpoint "+checkpoint_name
    model.save(save_path)

    if model_info['name'] == 'DQN':
        print("Saving buffer replay ")
        replay_path = model_info['checkpoints_path']+"replay "+checkpoint_name
        model.save_replay_buffer(replay_path)

    print("Saving outcomelog...")
    env.saveOutcomelog(path_name_binary=model_info['outcome_logs']+'/bin/'+checkpoint_name,
                    path_name_plain=model_info['outcome_logs']+'/plain/'+checkpoint_name)
    env.cleanOutcomeLog()