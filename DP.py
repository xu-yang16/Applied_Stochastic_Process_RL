import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from DP_lib import *

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': True},
    max_episode_steps=100,
    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
)

env = gym.make('FrozenLakeNotSlippery-v0', is_slippery=False)
S_n = env.observation_space.n
A_n = env.action_space.n

# obtain the optimal policy and optimal state-value function
policy_pi, V_pi, state_value = policy_iteration(env, 0.95)

print(state_value)
plt.scatter(range(1,5),state_value,marker='o',alpha=0.7,linewidths=0.01)
plt.xlabel('Episode')
plt.ylabel('Value of State 2')
plt.xticks([1,2,3,4])
plt.show()
# Optimal policy (pi)
# LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
plt.figure(figsize=(5, 16))
ax = sns.heatmap(policy_pi,  cmap="YlGnBu", annot=True, cbar=False, square=False)
ax.set_ylim([16, 0])
plt.show()
value = np.max(policy_pi,axis=1).reshape(4,4)
plt.figure(figsize=(8, 8))
ax = sns.heatmap(V_pi.reshape(4, 4),  cmap="YlGnBu", annot=True, cbar=False)
ax.set_ylim([4, 0])
plt.show()