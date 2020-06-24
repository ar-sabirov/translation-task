import torch.nn as nn
import torch

#from transforms import OneHotCharacters

class RNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        input_size = 130
        hidden_size = 128
        num_layers = 2
        self.ru_gru = nn.GRU(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        
        self.en_gru = nn.GRU(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        
        
        self.linear1 = nn.Linear(2 * hidden_size, 50)
        self.linear2 = nn.Linear(50, 1)
    
    def forward(self, ru, en):
        ru_gru, _ = self.ru_gru(ru)
        en_gru, _ = self.en_gru(en)
        
        
        ru_gru = ru_gru.squeeze()[:, -1, :]
        en_gru = en_gru.squeeze()[:, -1, :]
        
        grus_out = torch.cat([ru_gru, en_gru], dim=1)
        
        linear1 = self.linear1(grus_out)
        linear2 = self.linear2(linear1)
        
        output = torch.sigmoid(linear2)
        
        return output
        

if __name__ == "__main__":
    s = {'ru_name': 'Привет как дела', 'eng_name': 'This goes first', 'label': 1}
    oh = OneHotCharacters()
    inp = oh(s)
    
    
    n_letters = 127
    n_hidden = 128    
    n_categories = 2
    
    #hidden = torch.zeros(1, n_hidden)
    
    #rnn = RNN(n_letters, n_hidden, n_categories)
    rnn = RNN()
    
    output = rnn(inp['ru_name'], inp['eng_name'])
    