import torch.nn as nn
import torch
import pdb

class HSAM(nn.Module):
    def __init__(self, in_channels=20, reduction_ratio=8):
        super(HSAM, self).__init__()
            
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio if in_channels // reduction_ratio != 0 else 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio if in_channels // reduction_ratio != 0 else 1 , in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        out1 = self.avg_pool(x)
        out1 = out1.permute(0, 2, 1)
        out1 = self.relu(self.fc1(out1))
        avg_out = self.fc2(out1)
        
        out2 = self.max_pool(x)
        out2 = out2.permute(0, 2, 1)
        out2 = self.relu(self.fc1(out2))
        max_out = self.fc2(out2)

        out = torch.cat((avg_out , max_out), dim=1)
        out = self.conv_layer(out)
        # out = avg_out + max_out
        return self.sigmoid(out)
