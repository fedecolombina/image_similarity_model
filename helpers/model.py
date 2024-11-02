import torch.nn as nn
import torch.nn.functional as F

class SimilarityCNN(nn.Module):

    def __init__(self):
        super(SimilarityCNN, self).__init__()

        # Assume 200x200 gray-scale images as input
        self.convLayer1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.convLayer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.convLayer3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptivePool = nn.AdaptiveAvgPool2d((7, 7))

        self.fullyConnected1 = nn.Linear(64 * 7 * 7, 128)
        self.fullyConnected2 = nn.Linear(128, 64)

        self.dropout = nn.AlphaDropout(p=0.2) # Use AlphaDropout because of SELU

    def forward(self, x):

        x = self.pool(F.selu(self.convLayer1(x)))
        x = self.dropout(x)
        
        x = self.pool(F.selu(self.convLayer2(x)))
        x = self.dropout(x)

        x = self.pool(F.selu(self.convLayer3(x)))
        x = self.dropout(x)

        x = self.adaptivePool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = F.selu(self.fullyConnected1(x))
        x = self.dropout(x)
        x = self.fullyConnected2(x)
        
        return x