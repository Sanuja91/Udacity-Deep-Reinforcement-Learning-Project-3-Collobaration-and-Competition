from unityagents import UnityEnvironment
import numpy as np
import torch, time, os
import pandas as pd

def initialize_env(params):
    """Initialies the environment 
    Params
    ==========
    multiple_agents (boolean): multiple agents or single agent"""

    if params['offline']:
        env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe", worker_id = 1, no_graphics = params['no_graphics'])
    else:
        env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')


    """Resetting environment"""
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode = params['train_mode'])[brain_name]

    num_agents = len(env_info.agents)

    # number of agents in the environment
    print('Number of agents:', num_agents)

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    # print('States look like:', states[0])
    state_size = len(states[0])

    print('States have length:', state_size)
    print('States initialized:', len(states))
    print('Number of Agents:', num_agents)

    return env, env_info, states, state_size, action_size, brain_name, num_agents

class Seeds(object):
    def __init__(self, path):
        self.seeds = pd.read_csv(path, header=None, names=['seed'], index_col=None, dtype=np.int32)
        self.idx = 0
        
    def next(self):
        self.idx = self.idx + 1
        seed = self.seeds.seed.iloc[self.idx-1]
        return seed


# Checks if GPU is available else runs on CPU
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('\nSelected device = {}\n'.format(device))
    return device

# Prints a Break Comment for easy visibility on logs
def print_break(string):
    print("\n################################################################################################ " + string + "\n")

# Return normalized data
def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def update_csv(fileName, episode, average_score, max_score):
    """Updates a CSV file with train and test rewards"""
    COLUMN_NAMES = ["EPISODE", "AVERAGE SCORE", "MAX_SCORE"]  
    file_path = "results\\" + fileName + ".csv"
    average_score = float("{0:.2f}".format(average_score))
    max_score = float("{0:.2f}".format(max_score))
    if os.path.exists(file_path):
        df = pd.DataFrame(columns = COLUMN_NAMES)    
        df.set_index("EPISODE", inplace = True)
        df.at[episode] = np.array([average_score, max_score])
        prev_df = pd.read_csv(file_path)
        prev_df.set_index("EPISODE", inplace = True)
        df = pd.concat([prev_df, df])
        df.to_csv(file_path)
    else:
        df = pd.DataFrame(columns = COLUMN_NAMES)    
        df.set_index("EPISODE", inplace = True)
        df.at[episode] = np.array([average_score, max_score])
        df.to_csv(file_path)

# Get important data from file into a Dataframe
def open_json(relative_path):
    path = relative_path
    
    file_object  = open(path, "r")
    df = pd.read_json(file_object, orient='columns')
    
    if df.empty:
        print("{} is empty".format(relative_path))
        return 

    return df



