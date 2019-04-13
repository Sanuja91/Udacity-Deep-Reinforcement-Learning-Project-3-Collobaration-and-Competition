from collections import deque

import time

import numpy as np

import torch

from utilities import update_csv
from tensorboardX.writer import SummaryWriter


PARAMETER_NOISE = 0.01

def train(agents, params, num_processes):
    """Training Loop for value-based RL methods.
    Params
    ======
        agent (object) --- the agent to train
        params (dict) --- the dictionary of parameters
    """
    n_episodes = params['episodes']
    maxlen = params['maxlen']
    name = params['agent_params']['name']
    brain_name = params['brain_name']
    env = params['environment']
    add_noise = params['agent_params']['add_noise']
    pretrain = params['pretrain']
    pretrain_length = params['pretrain_length']
    num_agents = num_processes
    scores = np.zeros(num_agents)                     # list containing scores from each episode
    scores_window = deque(maxlen=maxlen)              # last N scores
    scores_episode = []
    writer = SummaryWriter(log_dir = params['log_dir'] + name)

    env_info = env.reset(train_mode = True)[brain_name]
    tic = time.time()
    timesteps = 0
    achievement_length = 0

    episode_start = 1
    if params['load_agent']:
        episode_start, timesteps = agents.load_agent()

    for i_episode in range(episode_start, n_episodes+1):
        tic = time.time()
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
 
        while True:
            states = torch.tensor(states)

            if pretrain and pretrain_length < len(agents.memory.memory):
                pretrain = False
     
            actions, noise_epsilon = agents.act(states, add_noise, pretrain = pretrain)
            
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished
            adjusted_rewards = np.array(env_info.rewards)

            if params['hack_rewards']:
                if adjusted_rewards[0] != 0:
                    adjusted_rewards[1] = adjusted_rewards[0] * params['alternative_reward_scalar']
                elif adjusted_rewards[1] != 0:
                    adjusted_rewards[0] = adjusted_rewards[1] * params['alternative_reward_scalar']

            actor_loss, critic_loss = agents.step(states, actions, adjusted_rewards, next_states, dones, pretrain = pretrain) 
            if actor_loss != None and critic_loss != None:

                if params['agent_params']['schedule_lr']:
                    actor_lr, critic_lr = agents.get_lr()
                else:
                    actor_lr, critic_lr = params['agent_params']['actor_params']['lr'], params['agent_params']['critic_params']['lr']

                writer.add_scalar('noise_epsilon', noise_epsilon, timesteps)
                writer.add_scalar('actor_loss', actor_loss, timesteps)
                writer.add_scalar('critic_loss', critic_loss, timesteps)
                writer.add_scalar('actor_lr', actor_lr, timesteps)
                writer.add_scalar('critic_lr', critic_lr, timesteps)

            print('\rTimestep {}\tMax: {:.2f}'.format(timesteps, np.max(scores)), end="")  

            scores += rewards                              # update the scores
            states = next_states                           # roll over the state to next time step
            if np.any(dones):                              # exit loop if episode finished
                break
                
            timesteps += 1 

            # Fills the buffer with experiences resulting from random actions 
            # to encourage exploration
            if timesteps % params['random_fill_every'] == 0:
                pretrain = True
                pretrain = params['pretrain_length']
            
        score = np.mean(scores)
        scores_episode.append(score)
        scores_window.append(score)       # save most recent score
     

        print('\rEpisode {}\tMax: {:.2f} \t Time: {:.2f}'.format(i_episode, np.max(scores), time.time() - tic), end="\n")
        
        if i_episode % params['save_every'] == 0:
            agents.save_agent(np.mean(scores_window), i_episode, timesteps, save_history = True)
        else:
            agents.save_agent(np.mean(scores_window), i_episode, timesteps, save_history = False)


        writer.add_scalars('scores', {'mean': np.mean(scores),
                                      'min': np.min(scores),
                                      'max': np.max(scores)}, timesteps)
                                        
        update_csv(name, i_episode, np.mean(scores), np.mean(scores))

        agents.step_lr(np.mean(scores))

        if np.mean(scores) > params['achievement']:
            achievement_length += 1
            if achievement_length > params['achievement_length']:
                toc = time.time()
                print("\n\n Congratulations! The agent has managed to solve the environment in {} episodes with {} training time\n\n".format(i_episode, toc-tic))
                writer.close()
                return scores
        else:
            achievement_length = 0

    writer.close()
    return scores


