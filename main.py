from unityagents import UnityEnvironment
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn

from agent import DDPGAgent, D4PGAgent
from train import train
from utilities import Seeds, initialize_env, get_device
from memory import NStepReplayBuffer

MULTI = True
device = get_device()                           # gets gpu if available

environment_params = {
    'multiple_agents': MULTI,                   # runs 20 or 1 arm environment
    'no_graphics': False,                       # runs no graphics windows version
    'train_mode': True,                         # runs in train mode
    'offline': True,                            # toggle on for udacity jupyter notebook
    'agent_count': 20 if MULTI else 1,  
    'device': device
}


env, env_info, states, state_size, action_size, brain_name, num_agents = initialize_env(environment_params)

seedGenerator = Seeds('seeds')
seedGenerator.next()

experience_params = {
    'seed': seedGenerator,                      # seed for the experience replay buffer
    'buffer_size': 300000,                      # size of the replay buffer
    'batch_size': 128,                         # batch size sampled from the replay buffer
    'rollout': 5,                               # n step rollout length    
    'agent_count': 2,  
    'gamma': 0.99,
    'device': device
}

experienceReplay = NStepReplayBuffer(experience_params)

params = {
    'episodes': 2000,                           # number of episodes
    'maxlen': 100,                              # sliding window size of recent scores
    'brain_name': brain_name,                   # the brain name of the unity environment
    'achievement': 30.,                         # score at which the environment is considered solved
    'environment': env,             
    'pretrain': True,                           # whether pretraining with random actions should be done
    'pretrain_length': 5000,                    # minimum experience required in replay buffer to start training 
    'random_fill': False,                       # basically repeat pretrain at specific times to encourage further exploration
    'random_fill_every': 10000,             
    'shape_rewards': False,                     # shapes 0 rewards into small negative reward
    'negative_reward': -0.0001,
    'log_dir': 'runs/',
    'load_agent': True,
    'agent_params': {
        'name': 'D4PG New Shape Rewards with Noise Decay',
        'experience_replay': experienceReplay,
        'device': device,
        'seed': seedGenerator,
        'num_agents': num_agents,               # number of agents in the environment
        'gamma': 0.99,                          # discount factor
        'tau': 0.0001,                          # mixing rate soft-update of target parameters
        'update_target_every': 350,             # update the target network every n-th step
        'update_every': 1,                     # update the active network every n-th step
        'actor_update_every_multiplier': 1,     # update actor every x timestep multiples of the crtic, critic needs time to adapt to new actor
        'update_intensity': 1,                 # learns from the same experiences several times
        'update_target_type': 'hard',           # should the update be soft at every time step or hard at every x timesteps
        'add_noise': True,                      # add noise using 'noise_params'
        'anneal_noise': True,  
        'schedule_lr': False,                   # schedule learning rates 
        'lr_steps': 30,                         # step iterations to cycle lr using cosine
        'lr_reset_every': 5000,                 # steps learning rate   
        'lr_reduction_factor': 0.9,             # reduce lr on plateau reduction factor
        'lr_patience_factor': 10,                # reduce lr after x (timesteps/episodes) not changing tracked item
        'actor_params': {                       # actor parameters
            'lr': 0.0005,                       # learning rate
            'state_size': state_size,           # size of the state space
            'action_size': action_size,         # size of the action space
            'seed': seedGenerator,              # seed of the network architecture
            'dropout': 0.05,
        },
        'critic_params': {                      # critic parameters
            'lr': 0.001,                         # learning rate
            'weight_decay': 3e-10,              # weight decay
            'state_size': state_size,           # size of the state space
            'action_size': action_size,         # size of the action space
            'seed': seedGenerator,              # seed of the network architecture
            'dropout': 0.05,
            'action_layer': True,
            'num_atoms': 75,
            'v_min': 0.0, 
            'v_max': 0.5
        },
        'ou_noise_params': {                    # parameters for the Ornstein Uhlenbeck process
            'mu': 0.,                           # mean
            'theta': 0.15,                      # theta value for the ornstein-uhlenbeck process
            'sigma': 0.2,                       # variance
            'seed': seedGenerator,              # seed
            'action_size': action_size
        },
        'ge_noise_params': {                    # parameters for the Gaussian Exploration process                   
            'max_epsilon': 0.3,                 
            'min_epsilon': 0.005,
            'decay_epsilon': True,    
            'patience_episodes': 2,         # episodes since the last best reward  
            'decay_rate': 0.95                   
        },
        
    }
}


agents = D4PGAgent(params=params['agent_params']) 

scores = train(agents=agents, params=params, num_processes=num_agents)

df = pd.DataFrame(data={'episode': np.arange(len(scores)), 'D4PG': scores})
df.to_csv('results/D4PG.csv', index=False)