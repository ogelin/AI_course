import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.autograd import Variable

from fashion import FashionMNIST
from secret_model import OneLayerModel, NLayerSigmoidModel

#####################################
#     PREPARATION DES DONNEES       #
#####################################

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

test_data = FashionMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))


# On va chercher 54000 (90%) indice au hasard parmis les 60000 disponible pour le training-set (donc on garde 10% pour le validation-set)
train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)
train_data.train_data = train_data.train_data[train_idx, :]
train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]

# Maintenant, allons chercher 6000 images (10%) qui seront utilise pour le validation set.
# Pourquoi est-ce qu'on va chercher les images sur le valid_data et non sur le train_data ?
mask = np.ones(60000)
mask[train_idx] = 0
valid_data.train_data = valid_data.train_data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.train_labels = valid_data.train_labels[torch.from_numpy(mask).type(torch.ByteTensor)]


# Creer des objets DataLoader qui precise a torch certain parametres tel que batch_size.
batch_size = 100
test_batch_size = 100
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size, shuffle=True)


# Pour afficher une image du train_data.
#plt.imshow(train_loader.dataset.train_data[1].numpy())
#plt.imshow(train_loader.dataset.train_data[10].numpy())



#####################################
#            FONCTIONS              #
#####################################
# A REGARDER
#Ici, FcNetwork derive de nn.Module
class FcNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# A REGARDER
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    return model


def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        valid_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print("valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return correct / len(valid_loader.dataset)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print("test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def experiment(model, epochs=10, lr=0.001):  #lr initial : 0.001
    best_precision = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    precisions = []
    for epoch in range(1, epochs + 1):
        model = train(model, train_loader, optimizer)
        precision = valid(model, valid_loader)
        precisions.append(precision)
        if precision > best_precision:
            best_precision = precision
            best_model = model

    #plt.plot(precisions, label="valid")
    #plt.xlabel("Epoch")
    #plt.ylabel("Precision")
    #plt.legend()
    #plt.show()

    return best_model, best_precision



#####################################
#            EXECUTION              #
#####################################

# A REGARDER
best_precision = 0
nNeurones = 512
for model in [NLayerSigmoidModel(nNeurones,10)]:
    print('\n' + "DEBUT DES TEST POUR LE MODEL:")
    model = model.cuda()
    model, precision = experiment(model, 100, 0.001)
    if precision > best_precision:
        best_precision = precision
        best_model = model

test(best_model, test_loader)
