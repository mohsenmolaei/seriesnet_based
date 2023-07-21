import torch
import torch.nn as nn
import torch.nn.functional as F
import HSAM
import pdb

if torch.cuda.is_available():
    # dev = "TPU:0"
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev) 

class LSTMModel(nn.Module):
    def __init__(self, features,features_c, hidden_dim, output_dim, n_layers):
        super(LSTMModel, self).__init__()
        
        self.input_dim = features+features_c
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # Define LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(self.input_dim, self.hidden_dim, 1, batch_first=True, dropout=0.2))
            else:
                self.lstm_layers.append(nn.LSTM(self.hidden_dim, self.hidden_dim, 1, batch_first=True, dropout=0.2))
        
        # Define HSAM layers
        self.hsam_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.hsam_layers.append(HSAM.HSAM(self.hidden_dim ))
        
        # Define output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, input_data, condition_data):
        
        batch_size = input_data.size(0)
        condition_data = condition_data.expand(batch_size, -1, -1)
        combined_data = torch.cat((input_data, condition_data), dim=2)
        # Initialize hidden and cell state
        h = [torch.zeros(1, combined_data.size(0), self.hidden_dim).to(device) for i in range(self.n_layers+1)]
        c = [torch.zeros(1, combined_data.size(0), self.hidden_dim).to(device) for i in range(self.n_layers+1)]
        
        # Run input sequence through LSTM layers
        lstm_out = combined_data
        for i in range(self.n_layers):
            lstm_out, (h[i+1], c[i+1]) = self.lstm_layers[i](lstm_out, (h[i], c[i]))
            
            if i > 0:

                hsam_out = self.hsam_layers[i-1](lstm_out)
                lstm_out = lstm_out * hsam_out
        
        # Apply output layer
        out = self.output_layer(lstm_out[:,-1,:])
        
        return out
