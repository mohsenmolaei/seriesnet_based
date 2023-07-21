import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio if in_channels // reduction_ratio != 0 else 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio if in_channels // reduction_ratio != 0 else 1 , in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).permute(0,2,1))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).permute(0,2,1))))
        out = torch.cat((avg_out , max_out), dim=1)
        out = self.conv_layer(out)
        return self.sigmoid(out.permute(0,2,1))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
      residual=x
      out = x * self.channel_att(x)
      out = out * self.spatial_att(out)
      return out + residual

class CBAM_Atten(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.cbam1 = CBAM(out_channels  , reduction_ratio, kernel_size)
        self.conv2 = nn.Conv1d(out_channels , out_channels, kernel_size=1)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        x = self.cbam1(x)
        # x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x
