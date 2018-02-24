import torch.nn as nn
import torch.nn.functional as F

class OneLayerModel(nn.Module):
    def __init__(self, nNeurones):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, nNeurones)
        self.fc2 = nn.Linear(nNeurones, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


class NLayerSigmoidModel(nn.Module):
    def __init__(self, nNeurones, nLayer):
        super().__init__()
        self.nLayer = nLayer
        self.fc1 = nn.Linear(28 * 28, nNeurones)
        self.fcX = nn.Linear(nNeurones, nNeurones)
        self.fc2 = nn.Linear(nNeurones, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        for i in range(0, self.nLayer-1):
            x = F.sigmoid(self.fcX(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


class NLayerSoftplusModel(nn.Module):
    def __init__(self, nNeurones, nLayer):
        super().__init__()
        self.nLayer = nLayer
        self.fc1 = nn.Linear(28 * 28, nNeurones)
        self.fcX = nn.Linear(nNeurones, nNeurones)
        self.fc2 = nn.Linear(nNeurones, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.softplus(self.fc1(x))
        for i in range(0, self.nLayer-1):
            x = F.softplus(self.fcX(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

class NLayerTanhModel(nn.Module):
    def __init__(self, nNeurones, nLayer):
        super().__init__()
        self.nLayer = nLayer
        self.fc1 = nn.Linear(28 * 28, nNeurones)
        self.fcX = nn.Linear(nNeurones, nNeurones)
        self.fc2 = nn.Linear(nNeurones, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.tanh(self.fc1(x))
        for i in range(0, self.nLayer-1):
            x = F.tanh(self.fcX(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

class NLayerRectifyModel(nn.Module):
    def __init__(self, nNeurones, nLayer):
        super().__init__()
        self.nLayer = nLayer
        self.fc1 = nn.Linear(28 * 28, nNeurones)
        self.fcX = nn.Linear(nNeurones, nNeurones)
        self.fc2 = nn.Linear(nNeurones, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        for i in range(0, self.nLayer-1):
            x = F.relu(self.fcX(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


class TwoLayerRectSoftplusModel(nn.Module):
    def __init__(self, nNeurones, nLayer):
        super().__init__()
        self.nLayer = nLayer
        self.fc1 = nn.Linear(28 * 28, nNeurones)
        self.fcX = nn.Linear(nNeurones, nNeurones)
        self.fc2 = nn.Linear(nNeurones, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.softplus(self.fcX(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

class TwoLayerSoftplusRectModel(nn.Module):
    def __init__(self, nNeurones, nLayer):
        super().__init__()
        self.nLayer = nLayer
        self.fc1 = nn.Linear(28 * 28, nNeurones)
        self.fcX = nn.Linear(nNeurones, nNeurones)
        self.fc2 = nn.Linear(nNeurones, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.softplus(self.fc1(x))
        x = F.relu(self.fcX(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

class ConvoModel(nn.Module):
    def __init__(self, nChannel, kernel_size):
        super(ConvoModel, self).__init__()
        self.conv1 = nn.Conv2d(1, nChannel, kernel_size)
        self.conv2 = nn.Conv2d(nChannel, nChannel, kernel_size)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2500, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x
