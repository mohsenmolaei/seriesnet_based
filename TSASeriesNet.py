import pdb
import torch
import torch.nn as nn
import Encoder_DARLM
import Decoder_DARLM 
import LSTM_block
import GRU_block
import ts_lstm
import ts_gru

class FullConnect(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullConnect, self).__init__()
        # self.fc1 = nn.Linear(in_features, 128)
        # self.dropout1 = nn.Dropout(p=0.6)
        # self.fc2 = nn.Linear(128, 64)
        # self.dropout2 = nn.Dropout(p=0.5)
        # self.fc3 = nn.Linear(64, 16)
        # self.dropout3 = nn.Dropout(p=0.4)
        self.fc4 = nn.Linear(in_features, out_features)
        self.relu= nn.ReLU()
    def forward(self, x):
        # x = self.dropout3(self.relu(self.fc3(self.dropout2(self.relu(self.fc2(self.dropout2(self.relu(self.fc1(x)))))))))
        x = self.fc4(x)
        return x
    
class ANNmodel(nn.Module): 
    def __init__(self,in_channels ,features_c ,features, output_num, num_levels_en, num_levels_de, kernel_size_EN,kernel_size_DE, dilation_c, hidden_size_lstm ,num_layers_lstm,num_layers_gru):
        super(ANNmodel, self).__init__()
        self.en_darlm = Encoder_DARLM.DDSTCN_block(in_channels ,features_c ,num_levels = num_levels_en ,kernel_size = kernel_size_EN, dilation_c=dilation_c)
        self.de_darlm = Decoder_DARLM.DDSTCN_block(in_channels ,features_c ,features, num_levels = num_levels_de ,kernel_size = kernel_size_DE, dilation_c=dilation_c)
        
        self.DA_lstm_layer = ts_lstm.DALSTM(features_c, hidden_size_lstm, hidden_size_lstm, in_channels, stateful_encoder=False, stateful_decoder=False)
        self.DA_gru_layer = ts_gru.DAGRU(features_c, hidden_size_lstm, hidden_size_lstm, in_channels, stateful_encoder=False, stateful_decoder=False)

        # self.attention_weights_cnn = nn.Linear(in_channels, 1)
        # self.attention_weights_lstm = nn.Linear(in_channels, 1)
        # self.attention_weights_gru = nn.Linear(in_channels, 1)

        self.Finalfc = FullConnect(in_channels, output_num)

        self.concat = nn.Linear(in_channels*3, in_channels)
        self.fc1 = nn.Linear((features_c+features)*2*in_channels, in_channels) 

    def forward(self, x , conditions):
        outEN = self.en_darlm(conditions)
        outDE = self.de_darlm(outEN,x)
        outCNN = outDE.reshape(outDE.size(0), -1)
        outCNN = self.fc1(outCNN)

        outDALSTM = self.DA_lstm_layer(conditions,x)
        outDAGRU  = self.DA_gru_layer(conditions,x)

        # pdb.set_trace()
        # calculate attention weights for each output
        # cnn_attention_weights = self.attention_weights_cnn(outCNN).squeeze(-1)
        # lstm_attention_weights = self.attention_weights_lstm(outDALSTM).squeeze(-1)
        # gru_attention_weights = self.attention_weights_gru(outDAGRU).squeeze(-1)

        # # apply softmax to the attention weights to get their relative importance
        # cnn_attention_weights = nn.functional.softmax(cnn_attention_weights, dim=-1)
        # lstm_attention_weights = nn.functional.softmax(lstm_attention_weights, dim=-1)
        # gru_attention_weights = nn.functional.softmax(gru_attention_weights, dim=-1)

        # # # apply attention weights tothe outputs
        # cnn_output = (outCNN * cnn_attention_weights.unsqueeze(-1))#.sum(dim=1)
        # lstm_output = (outDALSTM * lstm_attention_weights.unsqueeze(-1))#.sum(dim=1)
        # gru_output = (outDAGRU * gru_attention_weights.unsqueeze(-1))#.sum(dim=1)

        # concatenate the outputs with attention
        # concatenated_output = torch.cat((cnn_output, lstm_output, gru_output), dim=-1)

        concatenated_output = torch.cat(( outCNN, outDALSTM, outDAGRU), dim=-1)
        concatenated_output = self.concat(concatenated_output)

        return self.Finalfc(concatenated_output)