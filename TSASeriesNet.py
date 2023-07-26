import pdb
import torch
import torch.nn as nn
import Encoder_DARLM
import Decoder_DARLM 
import LSTM_block
import GRU_block
import ts_lstm
import ts_gru
import sfm_test


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
        # self.en_darlm = Encoder_DARLM.DDSTCN_block(in_channels ,features_c ,num_levels = num_levels_en ,kernel_size = kernel_size_EN, dilation_c=dilation_c)
        # self.de_darlm = Decoder_DARLM.DDSTCN_block(in_channels ,features_c ,features, num_levels = num_levels_de ,kernel_size = kernel_size_DE, dilation_c=dilation_c)
        # self.fc1 = nn.Linear((features_c+features)*2*in_channels, in_channels) 

        self.sfm = sfm_test.SFMSeq(input_size=1, hidden_size=1, freq_size=15, output_size=1, seq_len=hidden_size_lstm)
        # self.DA_lstm_layer = ts_lstm.DALSTM(features_c, hidden_size_lstm, hidden_size_lstm, in_channels, stateful_encoder=False, stateful_decoder=False)
        # self.DA_gru_layer = ts_gru.DAGRU(features_c, hidden_size_lstm, hidden_size_lstm, in_channels, stateful_encoder=False, stateful_decoder=False)


        # self.concat = nn.Linear(in_channels*3, in_channels)
        
        self.Finalfc = FullConnect(in_channels, output_num)

    def forward(self, x , conditions):
        # pdb.set_trace()
        # outEN = self.en_darlm(conditions)
        # outDE = self.de_darlm(outEN,x)
        # outCNN = outDE.reshape(outDE.size(0), -1)
        # outCNN = self.fc1(outCNN)

        # outDALSTM = self.DA_lstm_layer(conditions,x)
        # outDAGRU  = self.DA_gru_layer(conditions,x)

        # concatenated_output = torch.cat(( outCNN, outDALSTM, outDAGRU), dim=-1)
        # concatenated_output = self.concat(concatenated_output)
        sfm_out  = self.sfm(x)
        return self.Finalfc(sfm_out)