import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

env = gym.make("FrozenLake-v0")


# Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
# create lists to contain total rewards and steps per episode
# jList = []
rewardList = []
for index in range(num_episodes):
    # Reset environment and get first new observation
    observation = env.reset()
    rAll = 0
    done = False
    step = 0
    # The Q-Table learning algorithm
    while step < 99:
        step += 1
        # Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[observation, :] + np.random.uniform(1, env.action_space.n) * (1. / (index + 1)))
        # Get new state and reward from environment
        s1, r, done, _ = env.step(action)
        # Update Q-Table with new knowledge
        Q[observation, action] = Q[observation, action] + lr * (r + y * np.max(Q[s1, :]) - Q[observation, action])
        rAll += r
        observation = s1
        if done:
            break
    # jList.append(j)
    rewardList.append(rAll)

print("Score over time: " + str(sum(rewardList)/num_episodes))
print("Final Q-Table Values")
print(Q)
