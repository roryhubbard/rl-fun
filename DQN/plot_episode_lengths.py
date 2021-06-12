import matplotlib.pyplot as plt
import pickle
import numpy as np


plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['savefig.facecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams["figure.autolayout"] = True
# plt.rcParams['legend.facecolor'] = 'white'


with open("DQN/episode_rewards.txt", "rb") as f:
    episode_rewards = pickle.load(f)

averaged_episode_rewards = np.convolve(
    episode_rewards, np.ones(100)/100, mode='valid')

fig, ax = plt.subplots()
ax.plot(episode_rewards, label='true')
ax.plot(averaged_episode_rewards, label='averaged')

ax.legend()
plt.show()
plt.close()
