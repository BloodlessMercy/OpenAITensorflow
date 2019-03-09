import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')

bestLength = 0
episodeLengths = []
bestWeights = np.zeros(4)

for weightIndex in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)
    weightLength = []
    for trialIndex in range(100):
        observation = env.reset()
        done = False
        cnt = 0
        while not done:
            # Render the environment on the screen for monitoring;
            # Can be removed in later steps to speed up the learning process
            # env.render()
            # Increment count to keep track of total number of steps taken including termination step
            cnt += 1

            # Select a random action from the distribution of possible actions (action_space).
            # In this example -1 or 1, for left or right, respectively
            action = 1 if np.dot(observation, new_weights) > 0 else 0

            # Observation = The state of the environment after the step was taken
            # Reward = The total number of steps taken up until that point
            # Done = Flag variable declaring the game over or not
            observation, reward, done, _ = env.step(action)

            if done:
                break
        weightLength.append(cnt)
    averageLength = float(sum(weightLength) / len(weightLength))

    if averageLength > bestLength:
        bestLength = averageLength
        bestWeights = new_weights
    episodeLengths.append(averageLength)

    if weightIndex % 10 == 0:
        print('best length is: ', bestLength)
done = False
cnt = 0
env = wrappers.Monitor(env, './CartPole1/')
observation = env.reset()

while not done:
    # Render the environment on the screen for monitoring;
    # Can be removed in later steps to speed up the learning process
    env.render()
    # Increment count to keep track of total number of steps taken including termination step
    cnt += 1

    # Select a random action from the distribution of possible actions (action_space).
    # In this example -1 or 1, for left or right, respectively
    action = 1 if np.dot(observation, bestWeights) > 0 else 0

    # Observation = The state of the environment after the step was taken
    # Reward = The total number of steps taken up until that point
    # Done = Flag variable declaring the game over or not
    observation, reward, done, _ = env.step(action)

    if done:
        break

print('with best weights, game lasted: ', cnt, ' moves')