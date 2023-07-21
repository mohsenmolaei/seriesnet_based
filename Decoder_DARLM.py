# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 17:14:57 2023

@author: molae
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import pdb
from CBAM import CBAM_Atten

class DepthwiseNet_First(nn.Module):
    def __init__(self, in_channels, out_channels, features_c, features, num_levels, kernel_size, stride, dilation, padding ):
        super(DepthwiseNet_First, self).__init__()

        self.depthwise_bn = nn.BatchNorm1d(in_channels)
        self.Depthwise = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=out_channels)
        self.selu = nn.SELU()
        self.cbam = CBAM_Atten(in_channels= in_channels, out_channels=in_channels, reduction_ratio=8, kernel_size=3) 
        self.pointwise_conv = nn.Conv1d(features+ features_c + ((kernel_size-1) *2),features+features_c , kernel_size=1)
        self.linear = nn.Linear(in_channels, in_channels)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.linear.weight.data.normal_(0, 0.01) 
        
    def forward(self, x , conditions):
        outX = self.depthwise_bn(x)
        outX = self.Depthwise(outX)
        outC = self.depthwise_bn(conditions)
        outC = self.Depthwise(outC)
        concatenated = torch.cat((outC.transpose(1,2), outX.transpose(1,2)), dim=1)
        out = self.linear(concatenated).transpose(1,2)
        out = self.selu(out)
        out = (self.cbam(out))
        out =  (self.pointwise_conv(out.permute(0, 2, 1)).permute(0, 2, 1))
        
        concatenated = torch.cat((out.transpose(1,2) , x.transpose(1,2) , conditions.transpose(1,2)), dim=1)
        return self.linear(concatenated).transpose(1,2) #residual connection
     
class DepthwiseNet(nn.Module):
    def __init__(self, in_channels, out_channels, features,features_c, num_levels, kernel_size, stride, dilation, padding ):
        super(DepthwiseNet, self).__init__()
        self.features = features
        self.num_levels = num_levels
        self.depthwise_bn = nn.BatchNorm1d(in_channels)
        self.Depthwise = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=out_channels)
        self.selu = nn.SELU()


        self.cbam = CBAM_Atten(in_channels= in_channels, out_channels=in_channels, reduction_ratio=8, kernel_size=3) 
        self.pointwise_conv = nn.Conv1d((features+ features_c)*2 +((kernel_size-1) * (2**num_levels)),(features+features_c)*2  , kernel_size=1)
        self.linear = nn.Linear(in_channels, in_channels)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.linear.weight.data.normal_(0, 0.01) 
        
    def forward(self, x):
        out = self.depthwise_bn(x)
        out = self.Depthwise(out)
        out = self.selu(out)        
        out = self.cbam(out)
        out =  (self.pointwise_conv(out.permute(0, 2, 1)).permute(0, 2, 1))
        return self.linear(out.transpose(1,2) + x.transpose(1,2)).transpose(1,2) #residual connection
     
class DDSTCN_block(nn.Module):
    def __init__(self ,in_channels ,features_c ,features, num_levels, kernel_size=2, stride=0 ,dilation_c=2 ):
        super(DDSTCN_block, self).__init__()
        layers = []

        self.firstConv = nn.Conv1d(in_channels, in_channels , kernel_size=3, padding=1)
        self.pointwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            pad = (kernel_size-1) * dilation_size
            if l == 0:
                self.layer_1 = DepthwiseNet_First(in_channels, in_channels, features_c, features, l, kernel_size, stride=1, dilation=dilation_size, padding=pad)
            else:
                layers += [DepthwiseNet(in_channels, in_channels, features,features_c, l, kernel_size, stride=1, dilation=dilation_size, padding=pad)]
        self.network = nn.Sequential(*layers)

    def forward(self, conditions, x):
        outx = self.firstConv(x)
        out = self.layer_1(outx ,conditions)
        out = self.network(out)
        # out = self.pointwise_conv(out)
        return out
