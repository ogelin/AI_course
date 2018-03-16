import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.autograd import Variable
import random
from collections import namedtuple
import matplotlib.pyplot as plt
#from my_model import myAgent

#2 DQN.ipynb
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

#3 DQN.ipynb
# from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#4 DQN.ipynb
class myAgent(object):
    def __init__(self, gamma=0.99, batch_size=128):
        #Q pour actions
        #Target Q pour valeurs
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = gamma
        self.batch_size = 128
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.001)

    def act(self, x, epsilon=0.1):
        # if else avec epsilon qui est prob de choisir un action au hasard
        decision = random.random()
        if decision < epsilon:
            q = self.Q(x).type(torch.FloatTensor)
            qData = q.data
            qMax = qData.max(0)[1] # l'indice est en [1]
            res = Variable(torch.LongTensor(qMax))
            return res
        else:
            return Variable(torch.LongTensor([random.randrange(2)]))

    def backward(self, transitions):
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        state_action_values = self.Q(state_batch).gather(1, action_batch.view(-1,1))

        next_state_values = Variable(torch.zeros(128).type(torch.Tensor))
        next_state_values[non_final_mask] = self.Q(non_final_next_states).max(0)[1]



#5 DQN.ipynb
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    #memory buffer qui ajoute transitions qu'on a vues
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#6 DQN.ipynb
env = gym.make('CartPole-v0')
agent = myAgent()
memory = ReplayMemory(100000)
batch_size = 128

epsilon = 1
rewards = []

for i in range(5000):
    obs = env.reset()
    done = False
    total_reward = 0
    epsilon *= 0.99
    while not done:
        epsilon = max(epsilon, 0.1)
        obs_input = Variable(torch.from_numpy(obs).type(torch.FloatTensor))
        action = agent.act(obs_input, epsilon)
        #env.render() #Pour visualiser les mouvement a l'écran.
        tmp = action.data.numpy()
        next_obs, reward, done, _ = env.step(action.data.numpy()[0])
        memory.push(obs_input.data.view(1,-1), action.data,
                    torch.from_numpy(next_obs).type(torch.FloatTensor).view(1,-1), torch.Tensor([reward]),torch.Tensor([done]))
        obs = next_obs
        total_reward += reward
    rewards.append(total_reward)
    if memory.__len__() > 10000:
        # quand mémoire est pleine commencer à train l'agent avec backward
        #batch,current start, batch.nextstate
        batch = memory.sample(batch_size)
        agent.backward(batch)

pd.DataFrame(rewards).rolling(50, center=False).mean().plot()
plt.show()