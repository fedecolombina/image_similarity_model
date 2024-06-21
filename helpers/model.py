import torch.nn as nn
import torch.nn.functional as F

class myCNN(nn.Module):

    def __init__(self):
        super(myCNN, self).__init__()
        
        #convolutional layers (200x200 RGB images as input)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        #pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #fully connected layers. last pooling layer has [batch_size, 64, 50, 50]
        self.fc1 = nn.Linear(64 * 50 * 50, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):

        x = self.pool(F.selu(self.conv1(x)))
        x = self.pool(F.selu(self.conv2(x)))
        
        #flatten
        x = x.view(-1, 64 * 50 * 50)

        x = F.selu(self.fc1(x))
        x = self.fc2(x)
        
        return x