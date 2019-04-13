from collections import deque
import random
import torch
import numpy as np

class NStepReplayBuffer:
    """
    N step replay buffer to hold experiences for training. 
    Returns a random set of experiences without priority.

    The replay buffer can adapt to holding a rollout of experiences until it
    reaches the rollout length upon which the n step experience will be calculated
    using the rollout of experiences and store in memory as a single experience
    """
    def __init__(self, params):
        self.memory = deque(maxlen=params['buffer_size'])
        self.device = params['device']
        self.gamma = params['gamma']
        self.rollout_length = params['rollout_length']
        self.agent_count = params['agent_count']
        self.batch_size = params['batch_size']

        # Creates a deque to handle to store a rollout of experiences for each agent
        if self.rollout_length > 1:
            self.n_step = []  
            for _ in range(self.agent_count):
                self.n_step.append(deque(maxlen=self.rollout_length))

    def add(self, experience):
        """
        Checks if in n step or regular mode and acts accordingly
        If in the n step mode, it holds upto n experiences until the rollout length is reached following 
        which a discounted n step return is calculated (new reward)
        If in regular mode, it simply adds the experience to the replay buffer
        """
        states, actions, rewards, next_states, dones = experience
    
        # If rollouts > 1, its in n step mode
        if self.rollout_length > 1:
            for actor in range(self.agent_count):
                
                # Adds experience into n step deques
                self.n_step[actor].append((states[actor], actions[actor], rewards[actor], next_states[actor], dones[actor]))

            # Abort process over here if rollout length worth of experiences haven't been reached
            if len(self.n_step[0]) < self.rollout_length:
                return
            
            # Converts the collection of experiences for each agent into a single
            # n step experience for each agent
            self._create_nstep_experiences()
    
        else:
            for actor in range(self.agent_count):
                state = states[actor].float()
                action = torch.tensor(actions[actor]).float()
                reward = torch.tensor(rewards[actor]).float()
                next_state = torch.tensor(next_states[actor]).float()
                done = torch.tensor(dones[actor]).float()

                # Adds experience into n step trajectory
                self.memory.append((state, next_state, action, reward, done))

    def sample(self):
        """
        Return a sample of size of batch size as an experience tuple.
        """
        
        batch = random.sample(self.memory, k = self.batch_size)
        state, next_state, actions, rewards, dones = zip(*batch)

        # Stacks the experiences 
        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        dones = torch.stack(dones).to(self.device)

        return (state, next_state, actions, rewards, dones)

    def _create_nstep_experiences(self):
        """
        Takes a stack of experiences of the rollout length and calculates
        the n step discounted return as the new reward
        It takes the intial state and the state at the end of the rollout as 
        the n step state as well
  
        Returns are simply summed discounted rewards
        """

        # Unpacks and stores the experience tuples for each actor in the environment
        # from their respective n step deques
        for agent_experiences in self.n_step:

            states, actions, rewards, next_states, dones = zip(*agent_experiences)
            
            # The immediate reward is not discounted
            returns = rewards[0]

            # Every following reward is exponentially discounted by gamma
            # Gamma is the discounting factor can be used to control the value of future rewards
            for i in range(1, self.rollout_length):
                returns += self.gamma**i * rewards[i]
                if np.any(dones[i]):
                    break

            state = states[0].float()
            nstep_state = torch.tensor(next_states[i]).float()
            action = torch.tensor(actions[0]).float()
            done = torch.tensor(dones[i]).float()
            returns = torch.tensor(returns).float()
            self.memory.append((state, nstep_state, action, returns, done))

    def ready(self):
        return len(self.memory) > self.batch_size