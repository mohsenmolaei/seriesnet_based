import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from CBAM import CBAM_Atten

class DepthwiseNet(nn.Module):
    def __init__(self, in_channels, out_channels, features, num_levels, kernel_size, stride, dilation, padding):
        super(DepthwiseNet, self).__init__()
        self.depthwise_bn = nn.BatchNorm1d(in_channels)
        self.Depthwise = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=out_channels)
        self.selu = nn.SELU()
        self.cbam = CBAM_Atten(in_channels=in_channels, out_channels=in_channels, reduction_ratio=8, kernel_size=3)
        self.pointwise_conv = nn.Conv1d(features + ((kernel_size - 1) * (2 ** num_levels)), features, kernel_size=1)
        self.linear = nn.Linear(in_channels, in_channels)
        nn.init.normal_(self.linear.weight, 0, 0.01)

    def forward(self, x):
        out = self.depthwise_bn(x)
        out = self.Depthwise(out)
        out = self.selu(out)
        out = self.cbam(out)
        out = (self.pointwise_conv(out.permute(0, 2, 1)).permute(0, 2, 1))
        return self.linear(out.transpose(1,2) + x.transpose(1,2)).transpose(1,2) #residual connection


class DDSTCN_block(nn.Module):
    def __init__(self, num_inputs, features, num_levels, kernel_size=2, stride=0, dilation_c=2):
        super(DDSTCN_block, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_inputs
        self.firstConv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels)
        # self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        for l in range(num_levels):
            dilation_size = dilation_c ** l
            pad = (kernel_size - 1) * dilation_size
            layers += [DepthwiseNet(out_channels, out_channels, features, l, kernel_size, stride=1, dilation=dilation_size, padding=pad)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.firstConv(x)
        out = self.norm1(out)
        out = self.network(out)
        # out = self.pointwise_conv(out)
        return out