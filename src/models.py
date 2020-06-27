import torch.nn as nn
import torch.nn.functional as F
from torch import cat, sigmoid

class RNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 rnn_hidden_size: int,
                 rnn_num_layers: int,
                 fc_size: int,
                 output_size: int):
        nn.Module.__init__(self)

        self.rnn1 = nn.LSTM(input_size,
                                  rnn_hidden_size,
                                  rnn_num_layers,
                                  # bidirectional=True,
                                  batch_first=True)

        self.rnn2 = nn.LSTM(input_size,
                                  rnn_hidden_size,
                                  rnn_num_layers,
                                  # bidirectional=True,
                                  batch_first=True)

        self.linear1 = nn.Linear(2 * rnn_hidden_size, fc_size)
        self.linear2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        ru, en = x
        _, (ht1, _) = self.rnn1(ru)
        _, (ht2, _) = self.rnn2(en)

        grus_out = cat([ht1[-1], ht2[-1]], dim=1)

        linear1 = self.linear1(grus_out)
        linear2 = self.linear2(linear1)

        output = torch.sigmoid(linear2)

        return output


class RNNCNN(nn.Module):
    def __init__(self,
                 rnn_input_size: int = 71,
                 rnn_hidden_size: int = 64,
                 rnn_num_layers: int = 1):
        nn.Module.__init__(self)
        
        # self.rnn = nn.LSTM(rnn_input_size,
        #                          rnn_hidden_size,
        #                          rnn_num_layers,
        #                          batch_first=True)

        self.conv1 = nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2)

        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        #x, (h_n, c_n) = self.rnn(x)
        x = x[:, None, :, :]

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        return self.flatten(x)


class CompareModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.rnn_cnn1 = RNNCNN()
        self.rnn_cnn2 = RNNCNN()
        

        self.linear1 = nn.Linear(4096, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x):
        ru, en = x
        
        x_ru = self.rnn_cnn1(ru)
        x_en = self.rnn_cnn2(en)
        
        x = cat([x_ru, x_en], dim=1)
        
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        output = sigmoid(x)

        return output
    
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
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )
        
    def forward(self, x):
        return self.model(x)


class ChinatownModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.cnn1 = CNN()
        self.cnn2 = CNN()
        
        self.linear = nn.Linear(2048, 1)
        
    def forward(self, x):
        x_ru, x_en = x
        
        x_ru = x_ru[:, None, :, :]        
        x_ru = self.cnn1(x_ru)
        
        x_en = x_en[:, None, :, :]
        x_en = self.cnn2(x_en)
        
        x = cat([x_ru, x_en], dim=1)
        x = F.relu(x)

        x = self.linear(x)        
        
        return sigmoid(x)
        
        

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
