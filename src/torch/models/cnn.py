import torch.nn as nn
import torch.nn.functional as F
from torch import cat, sigmoid


class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=128,
                      kernel_size=7,
                      padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=7,
                      padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)


class ChinatownModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.cnn1 = CNN()
        self.cnn2 = CNN()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ru, x_en = x

        x_ru = self.cnn1(x_ru)
        x_en = self.cnn2(x_en)

        x = cat([x_ru, x_en], dim=1)

        x = self.fc_layers(x)

        return x
