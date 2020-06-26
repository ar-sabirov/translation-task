import torch


class RNN(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        input_size = 130
        hidden_size = 10
        num_layers = 1
        self.ru_gru = torch.nn.GRU(input_size,
                                   hidden_size,
                                   num_layers,
                                   batch_first=True)

        self.en_gru = torch.nn.GRU(input_size,
                                   hidden_size,
                                   num_layers,
                                   batch_first=True)

        self.linear1 = torch.nn.Linear(2 * hidden_size, 20)
        self.linear2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        ru, en = x
        ru_gru, _ = self.ru_gru(ru)
        en_gru, _ = self.en_gru(en)

        ru_gru = ru_gru.squeeze()[:, -1, :]
        en_gru = en_gru.squeeze()[:, -1, :]

        grus_out = torch.cat([ru_gru, en_gru], dim=1)

        linear1 = self.linear1(grus_out)
        output = self.linear2(linear1)

        #output = torch.sigmoid(linear2)

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
