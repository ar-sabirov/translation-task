import torch


class RNN(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 rnn_hidden_size: int,
                 rnn_num_layers: int,
                 fc_size: int,
                 output_size: int):
        torch.nn.Module.__init__(self)

        self.rnn1 = torch.nn.LSTM(input_size,
                                  rnn_hidden_size,
                                  rnn_num_layers,
                                  # bidirectional=True,
                                  batch_first=True)

        self.rnn2 = torch.nn.LSTM(input_size,
                                  rnn_hidden_size,
                                  rnn_num_layers,
                                  # bidirectional=True,
                                  batch_first=True)

        self.linear1 = torch.nn.Linear(2 * rnn_hidden_size, fc_size)
        self.linear2 = torch.nn.Linear(fc_size, output_size)

    def forward(self, x):
        ru, en = x
        _, (ht1, _) = self.rnn1(ru)
        _, (ht2, _) = self.rnn2(en)

        grus_out = torch.cat([ht1[-1], ht2[-1]], dim=1)

        linear1 = self.linear1(grus_out)
        linear2 = self.linear2(linear1)

        output = torch.sigmoid(linear2)

        return output


class RNNCNN(torch.nn.Module):
    def __init__(self,
                 rnn_input_size: int = 71,
                 rnn_hidden_size: int = 64,
                 rnn_num_layers: int = 1):
        torch.nn.Module.__init__(self)
        
        # self.rnn = torch.nn.LSTM(rnn_input_size,
        #                          rnn_hidden_size,
        #                          rnn_num_layers,
        #                          batch_first=True)

        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2)

        self.pool = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2)

        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        #x, (h_n, c_n) = self.rnn(x)
        x = x[:, None, :, :]

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        return self.flatten(x)


class CompareModel(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.rnn_cnn1 = RNNCNN()
        self.rnn_cnn2 = RNNCNN()
        

        self.linear1 = torch.nn.Linear(4096, 128)
        self.linear2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        ru, en = x
        
        x_ru = self.rnn_cnn1(ru)
        x_en = self.rnn_cnn2(en)
        
        x = torch.cat([x_ru, x_en], dim=1)
        
        x = self.linear1(x)
        x = self.linear2(x)
        
        output = torch.sigmoid(x)

        return output


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 6, 5)
#         self.pool = torch.nn.MaxPool2d(2, 2)
#         self.conv2 = torch.nn.Conv2d(6, 16, 5)
#         self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
