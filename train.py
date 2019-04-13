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
    achievement = params['achievement']
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
    best_min_score = 0.0
    timesteps = 0

    episode_start = 1
    if params['load_agent']:
        episode_start, timesteps = agents.load_agent()

    for i_episode in range(episode_start, n_episodes+1):
        timestep = time.time()
        env_info = env.reset(train_mode = True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)

        # if params['agent_params']['schedule_lr'] and timesteps % params['agent_params']['lr_reset_every'] == 0:
        #     agents.reset_lr()
        #     params['agent_params']['lr_reset_every'] *= 2   # increases lr reset duration
 
        while True:
            states = torch.tensor(states)

            if pretrain and pretrain_length < len(agents.memory.memory):
                pretrain = False
     
            actions, noise_epsilon = agents.act(states, add_noise, pretrain = pretrain)
            
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            # print("\n", rewards,"\n")
            dones = env_info.local_done                    # see if episode has finished
            adjusted_rewards = np.array(env_info.rewards)

            if params['shape_rewards']:
                adjusted_rewards[adjusted_rewards == 0] = params['negative_reward']
            # adjusted_rewards = torch.from_numpy(adjusted_rewards).to(device).float().unsqueeze(1)

            actor_loss, critic_loss = agents.step(states, actions, adjusted_rewards, next_states, dones, pretrain = pretrain) 
            if actor_loss != None and critic_loss != None:

                if params['agent_params']['schedule_lr']:
                    actor_lr, critic_lr = agents.get_lr()
                else:
                    actor_lr, critic_lr = params['agent_params']['actor_params']['lr'], params['agent_params']['critic_params']['lr']

                writer.add_scalar('noise_epsilon', noise_epsilon, timesteps)
                # writer.add_scalar('rewards', np.mean(rewards), timesteps)
                writer.add_scalar('actor_loss', actor_loss, timesteps)
                writer.add_scalar('critic_loss', critic_loss, timesteps)
                writer.add_scalar('actor_lr', actor_lr, timesteps)
                writer.add_scalar('critic_lr', critic_lr, timesteps)

            # if params['agent_params']['schedule_lr'] and timesteps % (params['agent_params']['lr_reset_every'] // params['agent_params']['lr_steps']) == 0:
            print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'.format(timesteps, np.mean(scores), np.min(scores), np.max(scores)), end="")  

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
        # print(scores)
        score = np.max(scores)            # the max of out of both scores is taken
        scores_episode.append(score)
        scores_window.append(score)       # save most recent score
     

        print('\rEpisode {}\tMax Score: {:.2f} \t Time: {:.2f}'.format(i_episode, np.max(scores), time.time() - timestep), end="\n")
        
        if i_episode % params['save_every'] == 0:
            agents.save_agent(np.mean(scores_window), i_episode, timesteps, save_history = True)
        else:
            agents.save_agent(np.mean(scores_window), i_episode, timesteps, save_history = False)


        writer.add_scalars('scores', {'mean': np.mean(scores),
                                      'min': np.min(scores),
                                      'max': np.max(scores)}, timesteps)
                                        
        update_csv(name, i_episode, np.mean(scores), np.mean(scores))

        agents.step_lr(np.mean(scores))
        if i_episode % 100 == 0:
            toc = time.time()
            print('\rEpisode {}\tAverage Score: {:.2f} \t Min: {:.2f} \t Max: {:.2f} \t Time: {:.2f}'.format(i_episode, np.mean(scores_window), np.min(scores_window), np.max(scores_window), toc - tic), end="")
        if np.mean(scores_window) >= achievement:
            toc = time.time()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} \t Time: {:.2f}'.format(i_episode-100, np.mean(scores_window), toc-tic))
            if best_min_score < np.min(scores_window):
                best_min_score = np.min(scores_window)
                # agents.save
                # for idx, a in enumerate(agents):
                #     torch.save(a.actor_local.state_dict(), 'results/' + str(idx) + '_' + str(i_episode) + '_' + name + '_actor_checkpoint.pth')
                #     torch.save(a.critic_local.state_dict(), 'results/' + str(idx) + '_' + str(i_episode) + '_' + name + '_critic_checkpoint.pth')
    writer.close()
    return scores