import torch.nn as nn
import torch.nn.functional as F
from torch import cat, sigmoid


class CNN(nn.Module):
    def __init__(self, out_channels: int):
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=out_channels,
                      kernel_size=7,
                      padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=7,
                      padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)


class ChinatownModel(nn.Module):
    def __init__(self, cnn_channels: int):
        nn.Module.__init__(self)
        self.cnn1 = CNN(cnn_channels)
        self.cnn2 = CNN(cnn_channels)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(3000, 3000),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(3000, 3000),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(3000, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ru, x_en = x

        x_ru = self.cnn1(x_ru)
        x_en = self.cnn2(x_en)

        x = cat([x_ru, x_en], dim=1)

        x = self.fc_layers(x)

        return x
