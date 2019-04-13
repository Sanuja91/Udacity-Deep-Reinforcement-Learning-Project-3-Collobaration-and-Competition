# coding: utf-8
from abc import ABCMeta, abstractmethod

import random
import copy
import os

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from models import Actor, Critic, D4PGCritic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, params):
        """Initialize an Agent object given a dictionary of parameters.
        
        Params
        ======
        * **params** (dict-like) --- a dictionary of parameters
        """
        pass

    @abstractmethod
    def act(self, state, action):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        * **state** (array_like) --- current state
        * **action** (array_like) --- the action values
        """
        pass

    @abstractmethod
    def step(self, states, actions, rewards, next_states, dones):
        """Perform a step in the environment given a state, action, reward,
        next state, and done experience.
        Params
        ======
        * **states** (torch.Variable) --- the current state
        * **actions** (torch.Variable) --- the current action
        * **rewards** (torch.Variable) --- the current reward
        * **next_states** (torch.Variable) --- the next state
        * **dones** (torch.Variable) --- the done indicator
        """
        pass

    @abstractmethod
    def learn_(self):
        """Update value parameters using given batch of experience tuples."""
        pass

    def _update_target_networks(self):
        """
        Updates the target networks using the active networks in either a 
        soft manner with the variable TAU or in a hard manner at every
        x timesteps
        """

        if self.update_target_type == "soft":
            self._soft_update(self.actor_active, self.actor_target)
            self._soft_update(self.critic_active, self.critic_target)
        elif self.update_target_type == "hard":
            self._hard_update(self.actor_active, self.actor_target)
            self._hard_update(self.critic_active, self.critic_target)

    def _soft_update(self, active, target):
        """
        Slowly updates the network using every-step partial network copies
        modulated by parameter TAU.
        """

        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(self.tau*param.data + (1-self.tau)*t_param.data)

    def _hard_update(self, active, target):
        """
        Fully copy parameters from active network to target network. To be used
        in conjunction with a parameter update_every that controls how many timesteps
        should pass between these hard updates.
        """

        target.load_state_dict(active.state_dict())

    def step_lr(self, score):
        """Steps the learning rate scheduler"""
        if self.schedule_lr:
            self.actor_scheduler.step(score)
            self.critic_scheduler.step(score)
            self.lr_steps += 1

        self.noise.decay(score)
        
    
    def get_lr(self):
        """Returns the learning rates"""
        actor_lr = 0
        critic_lr = 0
        for params in self.actor_optimizer.params:
            actor_lr =  params['lr']
        for params in self.critic_optimizer.params:
            critic_lr =  params['lr']
        return actor_lr, critic_lr


    def save_agent(self, average_reward, episode, timesteps, save_history = False):
        """Save the checkpoint"""
        checkpoint = {'actor_state_dict': self.actor_target.state_dict(), 'critic_state_dict': self.critic_target.state_dict(), 'average_reward': average_reward, 'episode': episode, 'timesteps': timesteps}
        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints") 
        
        filePath = 'checkpoints\\' + self.name + '.pth'
        torch.save(checkpoint, filePath)

        if save_history:
            filePath = 'checkpoints\\' + self.name + '_' + str(episode) + '.pth'
            torch.save(checkpoint, filePath)


    def load_agent(self):
        """Load the checkpoint"""
        filePath = 'checkpoints\\' + self.name + '.pth'

        if os.path.exists(filePath):
            checkpoint = torch.load(filePath, map_location = lambda storage, loc: storage)

            self.actor_active.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_active.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_state_dict'])

            average_reward = checkpoint['average_reward']
            episode = checkpoint['episode']
            timesteps = checkpoint['timesteps']
            
            print("Loading checkpoint - Average Reward {} at Episode {}".format(average_reward, episode))
            return episode + 1, timesteps
        else:
            print("\nCannot find {} checkpoint... Proceeding to create fresh neural network\n".format(self.name))   
            return 1, 0

    
class D4PGAgent(Agent):
    """An advance D4PG agent with an option to run on a simpler DDPG mode.
    The agent uses a distributional value estimation when running on D4PG vs
    the traditional single value estimation when running on DDPG mode."""
    
    def __init__(self, params):
        """Initialize an Agent object."""

        self.params = params
        self.update_target_every = params['update_target_every']
        self.update_every = params['update_every']
        self.actor_update_every_multiplier = params['actor_update_every_multiplier']
        self.update_intensity = params['update_intensity']
        self.gamma = params['gamma']
        self.action_size = params['actor_params']['action_size']
        self.num_agents = params['num_agents']
        self.num_atoms = params['critic_params']['num_atoms']
        self.v_min = params['critic_params']['v_min']
        self.v_max = params['critic_params']['v_max']
        self.update_target_type = params['update_target_type']
        self.device = params['device']
        self.name = params['name']
        self.lr_reduction_factor = params['lr_reduction_factor']
        self.tau = params['tau']
        self.d4pg = params['d4pg']

        # Distributes the number of atoms across the range of v min and max
        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)

        # Initialize time step count
        self.t_step = 0
        
        # Active and Target Actor networks
        self.actor_active = Actor(params['actor_params']).to(device)        
        self.actor_target = Actor(params['actor_params']).to(device)

        if self.d4pg:
            # Active and Target D4PG Critic networks
            self.critic_active = D4PGCritic(params['critic_params']).to(device)
            self.critic_target = D4PGCritic(params['critic_params']).to(device)
        else:
            # Active and Target Critic networks
            self.critic_active = Critic(params['critic_params']).to(device)
            self.critic_target = Critic(params['critic_params']).to(device)

        self.actor_optimizer = optim.Adam(self.actor_active.parameters(), lr = params['actor_params']['lr'])
        self.critic_optimizer = optim.Adam(self.critic_active.parameters(), lr = params['critic_params']['lr'])
        
        self.schedule_lr = params['schedule_lr']
        self.lr_steps = 0

        # Create learning rate schedulers if required to reduce the learning rate
        # depeninding on plateuing of scores
        if self.schedule_lr:
            self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, 
                                                    mode = 'max', 
                                                    factor = params['lr_reduction_factor'],
                                                    patience = params['lr_patience_factor'],
                                                    verbose = False,

                                                )
            self.critic_scheduler = ReduceLROnPlateau(self.critic_optimizer, 
                                                    mode = 'max',
                                                    factor = params['lr_reduction_factor'],
                                                    patience = params['lr_patience_factor'],
                                                    verbose = False,
                                                 )

        print("\n################ ACTOR ################\n")
        print(self.actor_active)
        
        print("\n################ CRITIC ################\n")
        print(self.critic_active)

        # Initiate exploration parameters by adding noise to the actions
        self.noise = params['noise']

        # Replay memory
        self.memory = params['experience_replay']

    def act(self, states, add_noise = True, pretrain = False):
        """Returns actions for given state as per current policy."""

        # If pretraining is active, the agent gives a random action thereby encouraging
        # intial exploration of the state space quickly
        if pretrain:
            actions = np.random.uniform(-1., 1., (self.num_agents, self.action_size))

        else:
            with torch.no_grad():
                actions = self.actor_active(states.to(device).float()).detach().to('cpu').numpy()
            if add_noise:
                noise = self.noise.create_noise(actions.shape)
                actions += noise
            
            actions = np.clip(actions, -1., 1.)        
        
        return actions, self.noise.epsilon


    def step(self, states, actions, rewards, next_states, dones, pretrain = False):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.memory.add((states, actions, rewards, next_states, dones))
        self.t_step += 1

        if pretrain == False:
            return self.learn_()
        
        return None, None

    def learn_(self):
        "Learns from experience using a distributional value estimation when in D4PG mode"
        actor_loss = None
        critic_loss = None
        
        # If enough samples are available in memory and its time to learn, then learn!
        if self.memory.ready() and self.t_step % self.update_every == 0:

            # Learns multiple times with the same set of experience
            for _ in range(self.update_intensity):

                # Samples from the replay buffer which has calculated the n step returns in advance
                # Next state represents the state at the n'th step
                states, next_states, actions, rewards, dones = self.memory.sample()

                if self.d4pg:
                    atoms = self.atoms.unsqueeze(0)

                    # Calculate log probability distribution using Zw with regards to stored actions
                    log_probs = self.critic_active(states, actions, log=True)

                    # Calculate the projected log probabilities from the target actor and critic networks
                    # Since back propogation is not required. Tensors are detach to increase speed
                    target_dist = self._get_targets(rewards, next_states).detach()

                    # The critic loss is calculated using a weighted distribution instead of the mean to
                    # arrive at a more accurate result. Cross Entropy loss is used as it is considered to 
                    # be the most ideal for categorical value distributions as utlized in the D4PG
                    critic_loss = -(target_dist * log_probs).sum(-1).mean()

                else:

                    # Get predicted next-state actions and Q values from target models
                    actions_next = self.actor_target(next_states)
                    Q_targets_next = self.critic_target(next_states, actions_next).detach()
                    # Compute Q targets for current states (y_i)
                    Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
                    # Compute critic loss
                    Q_expected = self.critic_active(states, actions)
                    critic_loss = F.mse_loss(Q_expected, Q_targets)

                # Execute gradient descent for the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_active.parameters(), 1)
                self.critic_optimizer.step()
                critic_loss = critic_loss.item()

                # Update actor every x multiples of critic
                if self.t_step % (self.actor_update_every_multiplier * self.update_every) == 0:
                    
                    if self.d4pg:
                        # Predicts the action for the actor networks loss calculation
                        predicted_action = self.actor_active(states)
                        # Predict the value distribution using the critic with regards to action predicted by actor
                        probs = self.critic_active(states, predicted_action)
                        # Multiply probabilities by atom values and sum across columns to get Q values
                        expected_reward = (probs * atoms).sum(-1)
                        # Calculate the actor network loss (Policy Gradient)
                        # Get the negative of the mean across the expected rewards to do gradient ascent
                        actor_loss = -expected_reward.mean()
                    else:
                        actions_pred = self.actor_active(states)
                        actor_loss = -self.critic_active(states, actions_pred).mean()

                    # Execute gradient ascent for the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()                    
                    actor_loss = actor_loss.item()

        # Updates the target networks every n steps
        if self.t_step % self.update_target_every == 0:
            self._update_target_networks()  
        
        # Returns the actor and critic losses to store on tensorboard
        return actor_loss, critic_loss       

    def _get_targets(self, rewards, next_states):
        """
        Calculate Yᵢ from target networks using the target actor and 
        and distributed critic networks
        """

        target_actions = self.actor_target(next_states)
        target_probs = self.critic_target(next_states, target_actions)

        # Project the categorical distribution
        projected_probs = self._get_value_distribution(rewards, target_probs)
        return projected_probs

    def _get_value_distribution(self, rewards, probs):
        """
        Returns the projected value distribution for the input state/action pair
        """

        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Rewards were stored with the first reward followed by each of the discounted rewards, sum up the 
        # reward with its discounted reward
        projected_atoms = rewards.unsqueeze(-1) + self.gamma**self.memory.rollout_length * self.atoms.unsqueeze(0)
        projected_atoms.clamp_(self.v_min, self.v_max)
        b = (projected_atoms - self.v_min) / delta_z

        # Professional level GPUs have floating point math that is more accurate 
        # to the n'th degree than traditional GPUs. This might be due to binary
        # imprecision resulting in 99.000000001 ceil() rounding to 100 instead of 99.
        # According to sources, forcibly reducing the precision seems to be the only
        # solution to the problem. Luckily it doesn't result in any complications to
        # the accuracy of calculating the lower and upper bounds correctly
        precision = 1
        b = torch.round(b * 10**precision) / 10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(self.device)

        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()

