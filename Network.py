from os import O_TEMPORARY
import torch
from sklearn.utils import shuffle
from torch.nn.modules import dropout


class Network(torch.nn.Module):

    def __init__(self, n_channels, n_classes, dropout_probability):

        super(Network, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.n_layers = 2
        self.hidden_dim = 70#80
        self.gru = torch.nn.GRU(input_size = n_channels, hidden_size = self.hidden_dim, num_layers = self.n_layers, batch_first = True, dropout = dropout_probability) 

        #self.lstm = torch.nn.LSTM(input_size = n_channels, hidden_size = self.hidden_dim, num_layers = self.n_layers, batch_first = True, dropout = dropout_probability) 
        self.dropout = torch.nn.Dropout(dropout_probability)
        self.fc1 = torch.nn.Linear(self.hidden_dim, n_classes)
        self.relu = torch.nn.ReLU()
    
        


    def forward(self, x):
        batch = x.size(0)
        h_0 = torch.zeros(self.n_layers, batch, self.hidden_dim, dtype=torch.float32)
        #c_0 = torch.zeros(self.n_layers, batch, self.hidden_dim, dtype=torch.float32)
        
        #output, h = self.gru(x, h)
        output,_ = self.gru(x,h_0)
        output = self.dropout(output)

        #output, _ = self.gru(x, (h_0, c_0))
        output = self.fc1(output)
        output = self.relu(output)
        output = output[:, -1, :]
       
        return output

 
