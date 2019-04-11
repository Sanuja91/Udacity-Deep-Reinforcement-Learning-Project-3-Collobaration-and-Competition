from collections import deque
import random
import torch
import numpy as np

class NStepReplayBuffer:
    """
    N step replay buffer to hold experiences for training. 
    Returns a random set of experiences without priority.

    The replay buffer can adapt to holding trajectories where 
    the buffer with hold SARS' data instead of hold a single experience
    state = state at t
    action = action at t
    reward = cumulative reward from t through t + n-1
    next_state = state at t + n
    where n = rollout length.
    """
    def __init__(self, params):
        # device, buffer_size=100000, gamma=0.99, rollout=5, agent_count=1
        self.memory = deque(maxlen=params['buffer_size'])
        self.device = params['device']
        self.gamma = params['gamma']
        self.rollout = params['rollout']
        self.agent_count = params['agent_count']
        self.batch_size = params['batch_size']

        # Creates a deque to handle nstep returns if in trajectory mode
        if self.rollout > 1:
            self.n_step = []
            # self.n_step = deque(maxlen=self.rollout)       
            for _ in range(self.agent_count):
                self.n_step.append(deque(maxlen=self.rollout))



    def add(self, experience):
        """
        Checks if in trajectory mode or experience mode. If in trajectory mode,
        it holds n step experiences until the rollout length is reached following 
        which a discounted n step return is calculated (new reward)
        """
        states, actions, rewards, next_states, dones = experience
        # print("\n\n#### INITIAL STATE", states.shape, "ACTION", actions.shape, "REWARD", rewards.shape, "\n\n")
        # If rollouts > 1, then its a trajectory not an episode
        if self.rollout > 1:
            for actor in range(self.agent_count):
                # print("\n\n#### NEXT STATE", states[actor].shape, "ACTION", actions[actor].shape, "REWARD", rewards[actor], "\n\n")
                # Adds experience into n step trajectory
                # print(dones)

                self.n_step[actor].append((states[actor], actions[actor], rewards[actor], next_states[actor], dones[actor]))

            # Abort process over here if trajectory length
            # worth of experiences haven't been reached
            if len(self.n_step[0]) < self.rollout:
                return
            
            # Converts the trajectory into and n step experience and adds to the buffer
            # (state, next_state, action, reward, done) = self._create_nstep_experiences()
            self._create_nstep_experiences()
    
            # state = state.float()
            # action = torch.tensor(action).float()
            # reward = torch.tensor(reward).float()
            # next_state = torch.tensor(next_state).float()
            # done = torch.tensor(done).float()
            # self.memory.append((state, next_state, action, reward, done))
        else:
            for actor in range(self.agent_count):
                # print("\n\n#### NEXT STATE", state.shape, "ACTION", action.shape, "REWARD", reward[actor], "\n\n")
                state = states[actor].float()
                action = torch.tensor(actions[actor]).float()
                reward = torch.tensor(rewards[actor]).float()
                next_state = torch.tensor(next_states[actor]).float()
                done = torch.tensor(dones[actor]).float()

                # Adds experience into n step trajectory
                self.memory.append((state, next_state, action, reward, done))
  
        # state, next_state, action, reward, done = experience
        # print("STATE", state.shape, "ACTION", action.shape, "REWARD", reward.shape)
        # state = state.float()
        # action = torch.tensor(action).float()
        # reward = torch.tensor(reward).float()
        # next_state = torch.tensor(next_state).float()
        # done = torch.tensor(done).float()
        # self.memory.append((state, next_state, action, reward, done))

    def sample(self):
        """
        Return a sample of size BATCH_SIZE as a tuple.
        """
        
        batch = random.sample(self.memory, k = self.batch_size)
        state, next_state, actions, rewards, dones = zip(*batch)

        # Transpose the num_agents and batch_size, for easy indexing later
        # e.g. from 64 samples of 2 agents each, to 2 agents with 64 samples
        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        dones = torch.stack(dones).to(self.device)

        # print("\n\nSTATES", state.shape, "ACTIONS", actions.shape, "REWARDS", rewards.shape, "\n\n")
        return (state, next_state, actions, rewards, dones)

    def _create_nstep_experiences(self):
        """
        Takes a stack of experiences nof the rollout length and calculates
        the n step discounted return as the new reward
        It takes the intial state and the state at the end of the rollout as 
        the next step and calculates the reward in terms of a discounted n step return
        Takes a stack of experience tuples of length rollout and calculates
        the n step 
        
        Returns are simply summed rewards
        """
        # Unpacks and stores the SARS' tuple for each actor in the environment
        # thus, each timestep actually adds K_ACTORS memories to the buffer,
        # for the Udacity environment this means 20 memories each timestep.
        for x, agent_experiences in enumerate(self.n_step):

            states, actions, rewards, next_states, dones = zip(*agent_experiences)
            
            # The immediate reward is not discounted
            returns = rewards[0]

            # Every following reward is exponentially discounted by gamma
            # Gamma can be used to control the value of future rewards
            for i in range(1, self.rollout):
                returns += self.gamma**i * rewards[i]
                if np.any(dones[i]):
                    # print("\n", dones)
                    # print(dones.shape)
                    # exit()
                    break

            state = states[0].float()
            nstep_state = torch.tensor(next_states[i]).float()
            action = torch.tensor(actions[0]).float()
            done = torch.tensor(dones[i]).float()
            returns = torch.tensor(returns).float()
            self.memory.append((state, nstep_state, action, returns, done))

        # return (state, nstep_state, action, returns, done)

    def ready(self):
        return len(self.memory) > self.batch_size