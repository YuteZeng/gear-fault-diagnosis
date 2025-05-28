import torch
import torch.nn as nn
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt


class conv1d_bn_relu_maxpool1(nn.Module):
    def __init__(self):
        super(conv1d_bn_relu_maxpool1, self).__init__()
        #residual function
        self.conv = nn.Conv1d(1, 16, kernel_size=100, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
    def forward(self, input): 
        out = self.conv(input) 
        out = self.bn(out) 
        out = self.relu(out) 
        out = self.drop(out)
        out = self.maxpool(out)
        return out 
class conv1d_bn_relu_maxpool2(nn.Module):
    def __init__(self):
        super(conv1d_bn_relu_maxpool2, self).__init__()
        #residual function
        self.conv = nn.Conv1d(16, 32, kernel_size=100, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
    def forward(self, input): 
        out = self.conv(input) 
        out = self.bn(out) 
        out = self.relu(out) 
        out = self.drop(out)
        out = self.maxpool(out)
        return out 


class MSCNN(nn.Module):
    def __init__(self, layers=[1, 1, 1, 1], num_classes=7):    #layers=[1, 1, 1, 1]
        self.inplanes3 = 1
        super(MSCNN, self).__init__()
        self.GAP1 = nn.AvgPool1d(kernel_size=2)
        self.GAP2 = nn.AvgPool1d(kernel_size=4)
        self.layer1 = conv1d_bn_relu_maxpool1()
        self.layer2 = conv1d_bn_relu_maxpool2()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc2 = nn.Linear(96, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.gap = nn.AdaptiveAvgPool1d((1))
    def forward(self, input):
        x0_1 = input
        x0_2 = self.GAP1(input)
        x0_3 = self.GAP2(input)

        x1 = self.layer1(x0_1)
        x1 = self.layer2(x1)
        x1 = self.gap(x1)

        x2 = self.layer1(x0_2)
        x2 = self.layer2(x2)
        x2 = self.gap(x2)

        x3 = self.layer1(x0_3)
        x3 = self.layer2(x3)
        x3 = self.gap(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        out = x.squeeze()
        out = self.relu(out)
        out1 = self.fc2(out)
        out2 = self.softmax(out1)
        return out1, out2

