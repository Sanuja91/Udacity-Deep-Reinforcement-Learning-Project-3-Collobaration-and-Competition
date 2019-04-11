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
from noise import OUNoise, GaussianExploration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""
    __metaclass__ = ABCMeta
    
    def __init__(self, params):
        """Initialize an Agent object given a dictionary of parameters.
        
        Params
        ======
        * **params** (dict-like) --- a dictionary of parameters
        """

        self.params = params
        self.tau = params['tau']
        
        # Replay memory: to be defined in derived classes
        self.memory = None
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    @abstractmethod
    def act(self, state, action):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        * **state** (array_like) --- current state
        * **action** (array_like) --- the action values
        """
        pass


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
        * **local_model** (PyTorch model) --- weights will be copied from
        * **target_model** (PyTorch model) --- weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
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
        * **betas** (float) --- a potentially tempered beta value for prioritzed replay sampling
        """
        pass

    @abstractmethod
    def learn_(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
        * **experiences** (Tuple[torch.Variable]) --- tuple of (s, a, r, s', done) tuples 
        """
        pass


class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    
    def __init__(self, params):
        """Initialize an Agent object.
        
        Params
        ======
            params (dict-like): dictionary of parameters for the agent
        """
        super().__init__(params)

        self.params = params
        self.update_every = params['update_every']
        self.gamma = params['gamma']
        self.num_agents = params['num_agents']
        self.update_target_type = params['update_target_type']
        self.name = "BATCH DDPG"
        self.update_target_every = params['update_target_every']
        self.update_every = params['update_every']
        self.update_intensity = params['update_intensity']
        
        # Actor Network (w/ Target Network)
        self.actor_active = Actor(params['actor_params']).to(device)
        self.actor_target = Actor(params['actor_params']).to(device)
        self.actor_optimizer = optim.Adam(self.actor_active.parameters(), lr=params['actor_params']['lr'])
        
        # Critic Network (w/ Target Network)
        self.critic_active = Critic(params['critic_params']).to(device)
        self.critic_target = Critic(params['critic_params']).to(device)

        print("\n################ ACTOR ################\n")
        print(self.actor_active)
        
        print("\n################ CRITIC ################\n")
        print(self.critic_active)


        self.critic_optimizer = optim.Adam(self.critic_active.parameters(),
                                           lr=params['critic_params']['lr'],
                                           weight_decay=params['critic_params']['weight_decay'])

        # Noise process
        self.noise = OUNoise(self.params['noise_params'])

        # Replay memory
        self.memory = params['experience_replay']
    
    def step(self, states, actions, rewards, next_states, dones, pretrain = False):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.memory.add((states, actions, rewards, next_states, dones))
        self.t_step += 1

        # Learn after done pretraining
        if pretrain == False:
            return self.learn()
        
        return None, None


    def act(self, states, add_noise = True, pretrain = False):
        """Returns actions for given state as per current policy."""

        # If pretraining is active, the agent gives a random action
        if pretrain:
            actions = np.random.uniform(-1., 1., (self.num_agents, self.action_size))

        else:
            with torch.no_grad():
                actions = self.actor_active(states.to(device).float()).detach().to('cpu').numpy()
                # print("\n\n################################# FRESH ACTIONS \n\n")
                # print(actions)
            if add_noise:
                noise = self.noise.create_noise(actions.shape)
                # print("\n\n################################# NOISE \n\n")
                # print(noise)
                actions += noise
                # print("\n\n################################# NOISY ACTIONS \n\n")
                # print(actions)
            
            actions = np.clip(actions, -1., 1.)        
            # print("\n\n################################# CLIPPED ACTIONS \n\n")
            # print(actions)

            # exit()
        
        return actions, self.noise.epsilon
    

    def learn(self):        
        # If enough samples are available in memory, get random subset and learn
        if self.memory.ready():
            return self.learn_()
        
        return None, None

        
        
    def learn_(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = self.memory.sample()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_active(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_active.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_active(states)
        actor_loss = -self.critic_active(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_active, self.critic_target)
        self.soft_update(self.actor_active, self.actor_target) 


    def save_agent(self, average_reward, episode, timesteps, save_history = False):
        """Save the checkpoint"""
        checkpoint = {'actor_state_dict': self.actor_target.state_dict(), 'critic_state_dict': self.critic_target.state_dict(), 'average_reward': average_reward, 'episode': episode, 'timesteps': timesteps}
        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints") 
        
        filePath = 'checkpoints\\' + self.name + '.pth'
        # print("\nSaving checkpoint\n")
        torch.save(checkpoint, filePath)

        if save_history:
            filePath = 'checkpoints\\' + self.name + '_' + str(episode) + '.pth'
            torch.save(checkpoint, filePath)


    def load_agent(self):
        """Load the checkpoint"""
        # print("\nLoading checkpoint\n")
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

    
    def _update_networks(self):
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
        Slowly updated the network using every-step partial network copies
        modulated by parameter TAU.
        """

        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(self.tau*param.data + (1-self.tau)*t_param.data)

    def _hard_update(self, active, target):
        """
        Fully copy parameters from active network to target network. To be used
        in conjunction with a parameter "C" that modulated how many timesteps
        between these hard updates.
        """

        target.load_state_dict(active.state_dict())

                    

class D4PGAgent(DDPGAgent):
    """Interacts with and learns from the environment."""
    
    def __init__(self, params):
        """Initialize an Agent object.
        
        Params
        ======
            params (dict-like): dictionary of parameters for the agent
        """
        # super().__init__(params)

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

        # Distributes the number of atoms across the range of v min and max
        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Actor Network (w/ Target Network)
        self.actor_active = Actor(params['actor_params']).to(device)        
        self.actor_target = Actor(params['actor_params']).to(device)

        ######################## D4PG ######################################
        # Critic Network (w/ Target Network)
        self.critic_active = D4PGCritic(params['critic_params']).to(device)
        self.critic_target = D4PGCritic(params['critic_params']).to(device)
        ######################## D4PG ######################################

        # ######################## DDPG ######################################
        # # Critic Network (w/ Target Network)
        # self.critic_active = Critic(params['critic_params']).to(device)
        # self.critic_target = Critic(params['critic_params']).to(device)
        # ######################## DDPG ######################################

        self.actor_optimizer = optim.Adam(self.actor_active.parameters(), lr = params['actor_params']['lr'])
        self.critic_optimizer = optim.Adam(self.critic_active.parameters(), lr = params['critic_params']['lr'])
        
        self.schedule_lr = params['schedule_lr']
        self.anneal_noise = params['anneal_noise']
        self.lr_steps = 0

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

        # Noise process
        self.noise = GaussianExploration(params['ge_noise_params'])

        # Replay memory
        self.memory = params['experience_replay']

    def step_lr(self, score):
        """Steps the learning rate scheduler"""
        if self.schedule_lr:
            self.actor_scheduler.step(score)
            self.critic_scheduler.step(score)
            self.lr_steps += 1

        if self.anneal_noise:
            self.noise.decay(score)
        
    
    def get_lr(self):
        """Returns the learning rates"""
        actor_lr = 0
        critic_lr = 0
        for param_group in self.actor_optimizer.param_groups:
            actor_lr =  param_group['lr']
        for param_group in self.critic_optimizer.param_groups:
            critic_lr =  param_group['lr']
        return actor_lr, critic_lr

    def learn_(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """        
        actor_loss = None
        critic_loss = None
        if self.t_step % self.update_every == 0:

            # Learns multiple times with the same set of experience
            for _ in range(self.update_intensity):

                
                # Samples from the replay buffer which has calculated the n step returns
                # Next state represents the state at the n'th step
                states, next_states, actions, rewards, dones = self.memory.sample()

                ######################## D4PG ######################################
                atoms = self.atoms.unsqueeze(0)

                # Calculate log probability distribution using Zw w.r.t. stored actions
                log_probs = self.critic_active(states, actions, log=True)

                # Calculate the projected log probabilities from the target actor and critic networks
                # Tensors are not required for backpropogation hence are detached for performance
                target_dist = self._get_targets(rewards, next_states).detach()

                # The critic loss is calculated using a weighted distribution instead of the mean to
                # arrive at a more accurate result. Cross Entropy loss is used as it is considered to 
                # be the most ideal for categorical value distributions as utlized in the D4PG
                critic_loss = -(target_dist * log_probs).sum(-1).mean()
                ######################## D4PG ######################################

                # ######################## DDPG ######################################
                # # Get predicted next-state actions and Q values from target models
                # actions_next = self.actor_target(next_states)
                # Q_targets_next = self.critic_target(next_states, actions_next).detach()
                # # Compute Q targets for current states (y_i)
                # Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
                # # Compute critic loss
                # Q_expected = self.critic_active(states, actions)
                # critic_loss = F.mse_loss(Q_expected, Q_targets)
                # ######################## DDPG ######################################

                # Execute gradient descent for the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_active.parameters(), 1)
                self.critic_optimizer.step()

                critic_loss = critic_loss.item()

                # Update actor every x multiples of critic
                if self.t_step % (self.actor_update_every_multiplier * self.update_every) == 0:
                    ######################## D4PG ######################################
                    # Predicts the action for the actor networks loss calculation
                    predicted_action = self.actor_active(states)

                    # Predict the value distribution using the critic with regards to action predicted by actor
                    probs = self.critic_active(states, predicted_action)

                    # Multiply probabilities by atom values and sum across columns to get Q values
                    expected_reward = (probs * atoms).sum(-1)

                    # Calculate the actor network loss (Policy Gradient)
                    # Get the negative of the mean across the expected rewards to do gradient ascent
                    actor_loss = -expected_reward.mean()
                    ######################## D4PG ######################################

                    # ######################## DDPG ######################################
                    # actions_pred = self.actor_active(states)
                    # actor_loss = -self.critic_active(states, actions_pred).mean()
                    # ######################## DDPG ######################################
                
                    # Execute gradient ascent for the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()                    
                    actor_loss = actor_loss.item()


        # Updates the target networks every n steps
        if self.t_step % self.update_target_every == 0:
            self._update_networks()  

        return actor_loss, critic_loss       

    def _get_targets(self, rewards, next_states):
        """
        Calculate Yᵢ from target networks using actor (πθ0)' and crtic (Zw')
        """

        target_actions = self.actor_target(next_states)
        target_probs = self.critic_target(next_states, target_actions)
        # Project the categorical distribution onto the supports
        projected_probs = self._categorical(rewards, target_probs)
        return projected_probs

    def _categorical(self, rewards, probs):
        """
        Returns the projected value distribution for the input state/action pair
        """

        # Create local vars to keep code more concise
        rewards = rewards.unsqueeze(-1)
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Rewards were stored with 0->(N-1) summed, take Reward and add it to
        # the discounted expected reward at N (ROLLOUT) timesteps
        projected_atoms = rewards + self.gamma**self.memory.rollout * self.atoms.unsqueeze(0)
        projected_atoms.clamp_(self.v_min, self.v_max)
        b = (projected_atoms - self.v_min) / delta_z

        # It seems that on professional level GPUs (for instance on AWS), the
        # floating point math is accurate to the degree that a tensor printing
        # as 99.00000 might in fact be 99.000000001 in the backend, perhaps due
        # to binary imprecision, but resulting in 99.00000...ceil() evaluating
        # to 100 instead of 99. Forcibly reducing the precision to the minimum
        # seems to be the only solution to this problem, and presents no issues
        # to the accuracy of calculating lower/upper_bound correctly.
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

