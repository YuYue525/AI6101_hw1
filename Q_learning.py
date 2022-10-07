from environment import CliffBoxPushingBase
from collections import defaultdict
import numpy as np
import random
import math

import matplotlib.pyplot as plt

class QAgent(object):
    def __init__(self):
        self.action_space = [1,2,3,4]
        # self.V = np.full((6, 14), -math.inf)
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.discount_factor=0.99
        self.alpha=0.5
        self.epsilon=0.01

    def take_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = self.action_space[np.argmax(self.Q[state])]
        return action

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state, action, next_state, reward):
        self.Q[state][self.action_space.index(action)] += self.alpha * (reward + self.discount_factor * self.Q[next_state][np.argmax(self.Q[next_state])] - self.Q[state][self.action_space.index(action)])
        
        

def plot_episode_rewards(episode_rewards, img_save_path):
    plt.cla()
    plt.scatter([x for x in range(len(episode_rewards))], episode_rewards, color = 'g', alpha = 0.1)
    
    plt.plot([x for x in range(len(episode_rewards))], [sum(episode_rewards[:(x+1)])/(x+1) for x in range(len(episode_rewards))], color = 'r', alpha = 0.7)
    
    plt.plot([x for x in range(len(episode_rewards)) if x > 0 and x % 100 == 0], [sum(episode_rewards[x-100:x])/100 for x in range(len(episode_rewards)) if x > 0 and x % 100 == 0], color = 'b', alpha = 0.7)
    
    plt.savefig(img_save_path)
    plt.cla()
    

if __name__ == '__main__':
    env = CliffBoxPushingBase()
    # you can implement other algorithms
    agent = QAgent()
    teminated = False
    rewards = []
    time_step = 0
    num_iterations = 10000
    
    episode_rewards = []
    
    for i in range(num_iterations):
        env.reset()
        while not teminated:
            state = env.get_state()
            action = agent.take_action(state)
            # print(action)
            reward, teminated, _ = env.step([action])
            next_state = env.get_state()
            rewards.append(reward)
            # print(f'step: {time_step}, actions: {action}, reward: {reward}')
            time_step += 1
            agent.train(state, action, next_state, reward)
        print(f'rewards: {sum(rewards)}')
        # print(f'print the historical actions: {env.episode_actions}')
        
        episode_rewards.append(sum(rewards))
        
        teminated = False
        rewards = []
    
    plot_episode_rewards(episode_rewards, "episode_rewards.png")
    # print(agent.Q[((5, 0), (4, 1))])
    
    # Q table testing
    f = open("test_log.txt", "w")
    test_step = 0
    env.reset()
    while not teminated:
        state = env.get_state()
        action = agent.action_space[np.argmax(agent.Q[state])]
        # print(action)
        reward, teminated, _ = env.step([action])
        next_state = env.get_state()
        rewards.append(reward)
        print(f'step: {test_step}, state: {state}, Q: {agent.Q[state]}, actions: {action}, next state: {next_state}, reward: {reward}')
        f.write(f'step: {test_step}, state: {state}, Q: {agent.Q[state]}, actions: {action}, next state: {next_state}, reward: {reward}\n')
        test_step += 1
        
    print(f'rewards: {sum(rewards)}')
    print(f'print the historical actions: {env.episode_actions}')
    f.write(f'\nrewards: {sum(rewards)}\n')
    f.write(f'print the historical actions: {env.episode_actions}\n')
    f.close()
        
