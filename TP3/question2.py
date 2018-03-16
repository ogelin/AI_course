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
        sample = random.random()
        if sample > epsilon:
            maxim = self.Q(x).data.max(0)[1]
            return Variable(torch.LongTensor(maxim))

        else:
            return Variable(torch.LongTensor([random.randrange(2)]))

    def backward(self, transitions):
        batch = Transition(*zip(*transitions))

        batch_state = Variable(torch.cat(batch.state))
        batch_action = Variable(torch.cat(batch.action))
        batch_reward = Variable(torch.cat(batch.reward))
        batch_next_state = Variable(torch.cat(batch.next_state))

        current_actions = self.Q(batch_state)
        current_q_values = current_actions.gather(1, batch_action.view(-1, 1))

        max_next_q_values = self.target_Q(batch_next_state).detach().max(1)[1]
        expected_q_values = (batch_reward.type(torch.FloatTensor) + (self.gamma * max_next_q_values.type(torch.FloatTensor))).view(-1, 1)

        assert current_q_values.shape == expected_q_values.shape  # les deux matrices doivent être de la même taille pour que ça marche

        loss = F.mse_loss(current_q_values, expected_q_values)

        soft_update(self.target_Q, self.Q, self.gamma)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()


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