[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Udacity : Deep Reinforcement Learning Nanodegree

## Project 3: Collaboration and Competition

![Trained Agent][image1]

For this project, we had to build an AI agent to conduct a episodic task for keeping the ball in play passing over the net from side to side without hitting the ball out of bounds or dropping it. 

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Implemented Agent
This project uses the latest state of the art D4PG agent algorithm to reach the start score. The agent is pretty sophisticated using distributed training, distributional value estimation, n-step bootstrapping, and an Actor-Critic architecture to provide fast, stable training in a continuous action space.

Additional tweaks such as batch normalization prior to Tanh action and negative reward shaping were implemented with tweaked n step replay buffer to gather the long term value of the actions.


## Installing the Agent
1. The environment is included in the repository with a Windows environment. The environmnet can also be downloaded from the following links for your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Run an environment of python 3.6.8 natively or with the use of Anaconda
3. Using pip install the `requirements.txt`

## Running the Agent

Simply start up the notebook and run each cell

## Solving the Environment

A single agent was implemented using the D4PG architecture to solve the environment in a parallel manner where the 8 state variables for each of the 2 agents are processed by the neural network in a batched manner in a parallelized fashion taking full advantage of the GPU power available and allowing the timesteps to proceed faster.