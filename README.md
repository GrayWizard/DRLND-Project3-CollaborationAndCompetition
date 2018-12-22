# Project 3: Collaboration and Competition

### Introduction

The goal of this project is to train an agent to operate in the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received (without discounting) are added up to get a score for each agent. This yields 2 (potentially different) scores. The maximum of these 2 scores yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Instructions

The project requires the installation of the environment provided by Udacity; see the detailed instructions [here](https://classroom.udacity.com/nanodegrees/nd893/parts/ec710e48-f1c5-4f1c-82de-39955d168eaa/modules/89b85bd0-0add-4548-bce9-3747eb099e60/lessons/3cf5c0c4-e837-4fe6-8071-489dcdb3ab3e/concepts/e85db55c-5f55-4f54-9b2b-d523569d9276). The following Python 3.5 libraries are required as well (if not provided by the Udacity DRLND environment): `unityagents`,`numpy`,`torch`,`matplotlib`.

After the enviromnent is set up and activated, run `python train_agent.py` to train the agent and `python demo_trained_agent.py` to see how the trained agent performs.