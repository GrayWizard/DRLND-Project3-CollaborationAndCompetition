# Note: this code is heavily based on the lecture DDPG code from Udacity DRN nanodegree
from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

# set up the environment
env = UnityEnvironment(file_name="d:\Courses\deep-reinforcement-learning\p3_collab-compet\Tennis_Windows_x86_64\Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# initialize the agent
agent = Agent(state_size=state_size, action_size=action_size, num_agents = num_agents, random_seed=42)

def ddpg(n_episodes=2000, max_t=1000, print_every=100):
    """Deep Deterministic Policy Gradients Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): frequency of printing/saving the state of the learning
    """    
    scores_list = []                           # list containing scores from each episode
    scores_window = deque(maxlen=print_every)  # last 'print_every' scores
    mean_scores = []                           # list of mean scores
    solved=False                               # has the environment been solved 
    max_score=0.0                              # current maximum score 

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations              # get the state
        scores = np.zeros(num_agents)                      # initialize the scores 
        agent.reset()                                      # reset the agent 
        for t in range(max_t):
            actions = agent.act(states)                    # get actions from the agents
            env_info = env.step(actions)[brain_name]       # send the actions to the environment
            next_states = env_info.vector_observations     # get the next set of states
            rewards = env_info.rewards                     # get the rewards
            done = env_info.local_done                     # check if done
            agent.step(states,actions,rewards,next_states,done)
            states = next_states
            scores += rewards                                # add the rewards
            if np.any(done):
                break
        score=np.max(scores)                   # score is the maximum score between two players 
        scores_window.append(score)            # save most recent score for averaging
        scores_list.append(score)              # save most recent score
        mean_score = np.mean(scores_window)    # calculate the mean score for the current window
        mean_scores.append(mean_score)         # save that mean score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
        # Save the learning state every 'print_every' episodes
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tMax Average Score: {:.2f}'.format(i_episode, np.mean(scores_window),max_score))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            # if a new maximum average score had been reached => save it and the network state
            if(max_score<mean_score):
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_max.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_max.pth')
                max_score=mean_score
        # If the desired score is achieved, save the learning state
        if mean_score>=0.5 and not(solved):
            solved=True
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_done.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_done.pth')
    return scores_list,mean_scores

# train
scores_list,mean_scores = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_list)+1), scores_list)
plt.plot(np.arange(1, len(mean_scores)+1), mean_scores)
plt.axhline(y=0.5, color='r', linestyle='-')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()