import torch.nn as nn
import torch.nn.functional as F

class SecretModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 2000)
        self.fc4 = nn.Linear(2000, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
