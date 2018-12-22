[//]: # (Image References)

[image1]: ./Figure_1.png "Rewards Plot"

# Project 3: Collaboration and Competition - Report

### Solution

#### Approach

My initial approach was to try my implementation of the DDPG learning algorithm from Project 2, slightly modified to work with a multi-agent environment. The training code (initially adapted from the lectures/Udacity's Deep Reinforcement Learning Github Repository) is contained in `train_agent.py` and the agent and network model in `ddpg_agent.py` and `model.py` accordingly.

#### Neural Network Architecture

##### Actor
The Actor network is a 4-layer fully connected neural network (with ReLu activations for the hidden layers and a Tanh activation for the output layer) with 24 units in the input layer, 128 units in each of the hidden layers and 4 units in the output layer. The output of the first hidden layer is batch-normalized for performance and stability improvement (based on a recommendation from the DRLND Slack channel).

##### Critic
The Critic network is a 4-layer fully connected neural network (with ReLu activations) with 24 units in the input layer, 128 units in each of the hidden layers and 1 unit in the output layer. The output of the first hidden layer is batch-normalized for performance and stability improvement (based on a recommendation from the DRLND Slack channel).

#### Hyperparameters
The initial approach used the following hyperparameters:

DDPG:
- n_episodes (int): maximum number of training episodes: *2000*
- max_t (int): maximum number of timesteps per episode: *1000*

Agent:
- BUFFER_SIZE = *int(1e6)*  # replay buffer size
- BATCH_SIZE minibatch size: *256*
- GAMMA: discount factor: *0.99*
- TAU: for soft update of target parameters: *1e-3*
- LR_ACTOR: learning rate of the actor: *2e-4*
- LR_CRITIC: learning rate of the critic: *2e-4*
- WEIGHT_DECAY: L2 weight decay: *0*

#### Results

After some experiments with the structure of actor and critic networks and learning rates, my DDPG implementation with parameters described above performed adequately and trained the agent to solve the environment with average score of +0.5 in 862 episodes.

After the environment was solved, I've let the training continue until all 2000 training episodes were completed. The learning continued to progress impressively well and reached an average score of +2.11 around episode 1900.

See the rewards plot below:

![Rewards Plot][image1]

The resulting stored weights for the initial point when the environment was solved (`checkpoint_critic_done.pth` and `checkpoint_actor_done.pth`) and the point with a maximum score (`checkpoint_critic_max.pth` and `checkpoint_actor_max.pth`) are stored in the project's repository.

Other network configurations (listed below) yielded somewhat worse results:
- Actor and Critic with 2 hidden layers of 64 neurons each - solved the environment in 1513 episodes
- Actor and Critic with 2 hidden layers of 256 neurons each - solved the environment in 1228 episodes
- Actor and Critic with 1st hidden layers of 64 neurons and 2nd hidden layer with 128 neurons - solved the environment in 1310 episodes
- Actor and Critic with 1st hidden layers of 128 neurons and 2nd hidden layer with 64 neurons - solved the environment in 1317 episodes

### Future work
As mentioned before, it seems that this DDPG implementation performs adequately. However, if we wanted to experiment further, another multi-agent training algorithm, such as PPO, A3C, or D4PG might be a way to go.