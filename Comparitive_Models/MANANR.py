import torch
import torch.nn as nn
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt



class BasicBlock1(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        #residual function
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, 2*out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(2*out_channel)
        self.NRRB1 = NRRB(out_channel, out_channel)
        self.NRRB2 = NRRB(2*out_channel, 2*out_channel)
    def forward(self, input): #(B,1,2048)
        out = self.conv1(input) #(B,32,2048)
        out = self.bn1(out) #(B,32,2048)
        out = self.NRRB1(out) #(B,32,2048)
        out = self.conv2(out) #(B,64,2048)
        out = self.bn2(out) #(B,64,2048)
        out = self.NRRB2(out) #(B,64,2048)
        return out #(B,64,2048)

class BasicBlock2(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock2, self).__init__()
        #residual function
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, 2*out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(2*out_channel)
        self.NRRB1 = NRRB(out_channel, out_channel)
        self.NRRB2 = NRRB(2*out_channel, 2*out_channel)
    def forward(self, input): #(B,1,2048)
        out = self.conv1(input) #(B,32,2048)
        out = self.bn1(out) #(B,32,2048)
        out = self.NRRB1(out) #(B,32,2048)
        out = self.conv2(out) #(B,64,2048)
        out = self.bn2(out) #(B,64,2048)
        out = self.NRRB2(out) #(B,64,2048)
        return out #(B,64,2048)

class BasicBlock3(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock3, self).__init__()
        #residual function
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, 2*out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(2*out_channel)
        self.NRRB1 = NRRB(out_channel, out_channel)
        self.NRRB2 = NRRB(2*out_channel, 2*out_channel)
    def forward(self, input): #(B,1,2048)
        out = self.conv1(input) #(B,32,2048)
        out = self.bn1(out) #(B,32,2048)
        out = self.NRRB1(out) #(B,32,2048)
        out = self.conv2(out) #(B,64,2048)
        out = self.bn2(out) #(B,64,2048)
        out = self.NRRB2(out) #(B,64,2048)
        return out #(B,64,2048)
    
class BasicBlock4(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock4, self).__init__()
        #residual function
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, 2*out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(2*out_channel)
        self.NRRB1 = NRRB(out_channel, out_channel)
        self.NRRB2 = NRRB(2*out_channel, 2*out_channel)
    def forward(self, input): #(B,1,2048)
        out = self.conv1(input) #(B,32,2048)
        out = self.bn1(out) #(B,32,2048)
        out = self.NRRB1(out) #(B,32,2048)
        out = self.conv2(out) #(B,64,2048)
        out = self.bn2(out) #(B,64,2048)
        out = self.NRRB2(out) #(B,64,2048)
        return out #(B,64,2048)

class NRRB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(NRRB, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.Shrinkage = Shrinkage(in_channel, out_channel, gap_size=(1))
    def forward(self, input): #(B,32,2048) 
        x = self.conv(input) #(B,32,2048)
        x = self.bn(x) #(B,32,2048)
        x = self.relu(x) #(B,32,2048)
        residual = x #(B,32,2048)
        x = self.conv(x) #(B,32,2048)
        x = self.bn(x) #(B,32,2048)
        x = self.relu(x) #(B,32,2048)
        x = self.Shrinkage(x)
        x += residual
        return x

class Shrinkage(nn.Module): 
    def __init__(self, in_channel, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.map = nn.MaxPool1d(kernel_size=64, stride=64)
        self.DownChannel = nn.Linear(2*channel, in_channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),)
    def forward(self, x): #(B,32,2048)
        #GAP
        x_abs = torch.abs(x)
        x_raw = x
        x_GAP = torch.abs(x)
        x_GAP = self.gap(x)
        x_GAP = torch.flatten(x_GAP , 1)
        average_GAP = x_GAP
        x_GAP = self.fc(x_GAP)
        x_GAP = torch.mul(average_GAP, x_GAP)
        #MAP
        x_MAP = torch.abs(x)
        x_MAP = self.map(x) 
        x_MAP = self.gap(x_MAP)
        x_MAP = torch.flatten(x_MAP, 1)
        average_MAP = x_MAP
        x_MAP = self.fc(x_MAP)
        x_MAP = torch.mul(average_MAP, x_MAP)
        #GAP + MAP
        x_ALL = torch.cat([x_GAP, x_MAP], dim=1)
        x_ALL = self.DownChannel(x_ALL)
        x_ALL = x_ALL.unsqueeze(2)
        sub = x_abs - x_ALL
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        output = torch.mul(torch.sign(x_raw), n_sub)
        return output

class Self_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.query = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.value = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input):
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).permute(0, 2, 1)
        #input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input)
        #input: B, C, H, W -> v: B, C, H * W
        v = self.value(input)
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)
 
        return self.gamma * out + input

class MANANR(nn.Module):
    def __init__(self, layers=[1, 1, 1, 1], num_classes=7):    #layers=[1, 1, 1, 1]
        self.inplanes3 = 1
        super(MANANR, self).__init__()
        # self.downsampleconv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, bias=None)
        # self.downsampleconv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=4, padding=1, bias=None)
        # self.downsampleconv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=8, padding=1, bias=None)
        self.layer1 = BasicBlock1(in_channel=1, out_channel=32, stride=1, downsample=None)
        self.layer2 = BasicBlock2(in_channel=1, out_channel=32, stride=1, downsample=None)
        self.layer3 = BasicBlock3(in_channel=1, out_channel=32, stride=1, downsample=None)
        self.layer4 = BasicBlock4(in_channel=1, out_channel=32, stride=1, downsample=None)
        self.sa = Self_Attention()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc1 = nn.Linear(64*4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x0):
        x1 = self.layer1(x0)
        x2 = self.layer2(x0)
        x3 = self.layer3(x0)
        x4 = self.layer4(x0)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x= self.sa(x)
        x = self.avg_pool(x)
        out = x.squeeze()
        out = self.fc1(out)
        out = self.tanh(out)
        out1 = self.fc2(out)
        out2 = self.softmax(out1)
        return out1, out2

