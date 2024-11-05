import torch.nn as nn
import torch.nn.functional as F

class SimilarityCNN(nn.Module):

    def __init__(self):
        super(SimilarityCNN, self).__init__()

        self.convLayer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.convLayer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.convLayer3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.convLayer4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256) 

        self.convLayer5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptivePool = nn.AdaptiveAvgPool2d((6, 6))

        self.fullyConnected1 = nn.Linear(512 * 6 * 6, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fullyConnected2 = nn.Linear(512, 256)

        self.dropoutCL = nn.Dropout(p=0.1) 
        self.dropoutFC = nn.Dropout(p=0.3) 


    def forward(self, x):

        x = self.convLayer1(x)
        x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.dropoutCL(x)

        x = self.convLayer2(x)
        x = self.bn2(x)
        x = self.pool(F.relu(x))
        x = self.dropoutCL(x)

        x = self.convLayer3(x)
        x = self.bn3(x)
        x = self.pool(F.relu(x))
        x = self.dropoutCL(x)

        x = self.convLayer4(x)
        x = self.bn4(x)
        x = self.pool(F.relu(x))
        x = self.dropoutCL(x)

        x = self.convLayer5(x)
        x = self.bn5(x)
        x = self.pool(F.relu(x))
        x = self.dropoutCL(x)

        x = self.adaptivePool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fullyConnected1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropoutFC(x)
        x = self.fullyConnected2(x)
        
        x = F.normalize(x, p=2, dim=1)
        
        return x