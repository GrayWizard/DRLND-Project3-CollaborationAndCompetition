# Note: this code is heavily based on the lecture DDPG code from Udacity DRN nanodegree
from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

# set up the environment
env = UnityEnvironment(file_name='d:\Courses\deep-reinforcement-learning\p3_collab-compet\Tennis_Windows_x86_64\Tennis.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# initialize the agent
agent = Agent(state_size=24, action_size=2, num_agents=2, random_seed=42)

# load the weights from files
agent.actor_local.load_state_dict(torch.load('checkpoint_actor_max.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic_max.pth'))

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
states = env_info.vector_observations               # get the current states
scores = np.zeros(2)                                # initialize the scores
while True:
    actions = agent.act(states)                     # get actions from the agent
    env_info = env.step(actions)[brain_name]        # send the actions to the environment
    states = env_info.vector_observations           # get the next set of states
    rewards = env_info.rewards                      # get the rewards
    done = np.any(env_info.local_done)              # see if episode has finished
    scores += rewards                               # update the score
    print('\rCurrent Score:\tAgent 1:{:.2f}\tAgent 2:{:.2f}'.format(scores[0],scores[1]), end="") # print current score
    if done:                                        # exit loop if episode finished
        break
print('\nMaximum score: {:.2f}'.format(np.max(scores)))
env.close()