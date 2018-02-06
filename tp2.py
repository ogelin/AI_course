import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from fashion import FashionMNIST
import torchvision.transforms as transforms
from torch import nn
from torch import optim


train_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

valid_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# On va chercher 54000 indice au hasard parmis les 60000 disponible pour le training-set (donc on garde 10% pour le validation-set)
train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)
train_data.train_data = train_data.train_data[train_idx, :]

# import pdb; pdb.set_trace()

x = torch.from_numpy(train_idx)
y = x.type(torch.LongTensor)

train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]

print(train_idx)