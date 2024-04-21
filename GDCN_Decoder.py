import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from CBAM import CBAM_Atten
from torch.nn.parameter import Parameter
import math
import pdb

class GDCN(nn.Module):
    def __init__(self,input_size, hidden_size ):
        super(GDCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size,input_size)

        self.set_parameters()
        self.reset_parameters()

    def reset_parameters(self):
      stdv = 1.0 / math.sqrt(self.hidden_size)
      for weight in self.parameters():
          weight.data.uniform_(-stdv, stdv)

    def set_parameters(self):
      # gate weights
      # W -> x, U -> h_last
      self.z_W = Parameter(torch.randn(self.input_size,self.hidden_size))
      self.z_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
      self.z_b = Parameter(torch.randn(self.hidden_size))

      self.r_W = Parameter(torch.randn(self.input_size,self.hidden_size))
      self.r_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
      self.r_b = Parameter(torch.randn(self.hidden_size))
      
      self.h_W = Parameter(torch.randn(self.input_size,self.hidden_size))
      self.h_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
      self.h_b = Parameter(torch.randn(self.hidden_size))

    def forward(self, input,h):
      z_t   = self.sigmoid(torch.matmul(input,self.z_W)+torch.matmul(h,self.z_U)+self.z_b)
      r_t   = self.sigmoid(torch.matmul(input,self.r_W)+torch.matmul(h,self.r_U)+self.r_b)
      h_hat = self.tanh(torch.matmul(input,self.h_W)+torch.matmul((torch.mul(r_t,h)),self.h_U)+self.h_b)
      h_t   = (torch.mul(z_t,h_hat) + torch.mul((1-z_t),h))
      out = self.output(h_t)
      return out

class DepthwiseNet_First(nn.Module):
    def __init__(self, in_channels, out_channels, features_c, features, num_levels, kernel_size, stride, dilation, padding ):
        super(DepthwiseNet_First, self).__init__()

        self.Depthwise = nn.Conv1d(features, features, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=features)
        self.Depthwise_condition = nn.Conv1d(features_c, features_c, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=features_c)
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.pointwise_conv = nn.Conv1d(in_channels + ((kernel_size - 1) * (2 ** num_levels)),in_channels , kernel_size=1)
        self.GDCN = GDCN(in_channels,in_channels)
        self.linear = nn.Linear(in_channels, in_channels)

        # self.init_weights()
    # def init_weights(self):
    #     """Initialize weights"""
    #     self.linear.weight.data.normal_(0, 0.01) 

    def forward(self, x , conditions):
      
      outX = self.Depthwise(x.permute(0, 2, 1))
      outC = self.Depthwise_condition(conditions.permute(0, 2, 1))
      concatenated = torch.cat((outC, outX), dim=1)
      out = self.selu(concatenated)
      out = self.pointwise_conv(out.permute(0, 2, 1))#.permute(0, 2, 1)
      out = self.GDCN(x.transpose(1,2),out.transpose(1,2)).transpose(1,2)
    #   return self.linear(out.transpose(1,2)).transpose(1,2)
      return out
     
class DepthwiseNet(nn.Module):
    def __init__(self, in_channels, out_channels, features,features_c, num_levels, kernel_size, stride, dilation, padding ):
        super(DepthwiseNet, self).__init__()
        # self.features = features
        self.num_levels = num_levels
        self.Depthwise = nn.Conv1d(features+features_c, features+features_c, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=features+features_c)
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.pointwise_conv = nn.Conv1d(in_channels + ((kernel_size - 1) * (2 ** num_levels)),in_channels  , kernel_size=1)
        # self.depthwise_bn = nn.BatchNorm1d(in_channels)
        # self.linear = nn.Linear(in_channels, in_channels)
        self.GDCN = GDCN(in_channels,in_channels)

    def forward(self, x):
      
      out = self.Depthwise(x.permute(0, 2, 1))
      out = self.selu(out)  
      out = self.pointwise_conv(out.permute(0, 2, 1))
      out = self.GDCN(x.transpose(1,2),out.transpose(1,2)).transpose(1,2)
      return out #self.linear(out.transpose(1,2)).transpose(1,2) #residual connection
     
class DDSTCN_block(nn.Module):
    def __init__(self ,in_channels ,features_c ,features, num_levels, kernel_size=2, stride=0 ,dilation_c=2 ):
        super(DDSTCN_block, self).__init__()
        layers = []

        self.firstConv = nn.Conv1d(in_channels, in_channels , kernel_size=3, padding=1)
        # self.pointwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
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
     
class DDSTCN_block(nn.Module):
    def __init__(self ,in_channels ,features_c ,features, num_levels, kernel_size=2, stride=0 ,dilation_c=2 ):
        super(DDSTCN_block, self).__init__()
        layers = []

        # self.firstConv = nn.Conv1d(in_channels, in_channels , kernel_size=3, padding=1)
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            pad = (kernel_size-1) * dilation_size
            if l == 0:
                self.layer_1 = DepthwiseNet_First(in_channels, in_channels, features_c, features, l, kernel_size, stride=1, dilation=dilation_size, padding=pad)
            else:
                layers += [DepthwiseNet(in_channels, in_channels, features,features_c, l, kernel_size, stride=1, dilation=dilation_size, padding=pad)]
        self.network = nn.Sequential(*layers)

    def forward(self, conditions, x):
      
    #   outx = self.firstConv(x)
      out = self.layer_1(x ,conditions)
      out = self.network(out)
      return out
