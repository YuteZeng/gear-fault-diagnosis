import torch
import torch.nn as nn
from torch import erf
import matplotlib.pyplot as plt




def new_soft_thresholding(x, t, h):
    return x + (1/2)*(((x-t)**2 + h)**0.5 - ((x+t)**2 + h)**0.5)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride):
        super(BasicBlock, self).__init__() 
        self.multiscaleCNN = Multiscale_CNN(in_channel, out_channel, stride)
        self.singlescaleCNN = Singlescale_CNN(in_channel, out_channel, stride)
        self.shrinkage_gap = Shrinkage_Gap(4*out_channel, gap_size=(1))
        self.shrinkage_map = Shrinkage_Map(4*out_channel, gap_size=(1))
        self.reshapeCNN = nn.Conv1d(8*out_channel, 4*out_channel, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)  #(64,16,1024)信号特征可视化时去掉
        self.stride = stride

    def forward(self, x): #(64,4,2048)
        out = self.multiscaleCNN(x) #(64,16,1024)
        # out = self.singlescaleCNN(x)
        residual = out #(64,16,1024)
        out1 = self.shrinkage_gap(out) #(64,16,1024)
        out2 = self.shrinkage_map(out) #(64,16,1024)
        out = torch.cat([out1, out2], dim=1) #(64,32,1024)
        out = self.reshapeCNN(out) #(64,16,1024)
        out += residual #(64,16,1024)
        # out = self.relu(out) #(64,16,1024)
        return out

class Self_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.query = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.value = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1)
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

class Multiscale_CNN(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Multiscale_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(out_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=stride)
        self.conv4 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.sa = Self_Attention()
    def forward(self, input):
        x1 = self.conv1(input) #(64,4,1024)
        x2_1 = self.conv1(input) #(64,4,1024)
        x2_2 = self.conv2(x2_1) #(64,4,1024)
        x3_1 = self.conv1(input) #(64,4,1024)
        x3_2 = self.conv2(x3_1) #(64,4,1024)
        x3_3 = self.conv3(x3_2) #(64,4,1024)
        x4_1 = self.maxpool(input) #(64,4,1024)
        x4_2 = self.conv4(x4_1) #(64,4,1024)
        x = torch.cat([x1, x2_2, x3_3, x4_2], dim=1) #(64,16,1024)
        x = self.sa(x)
        return x
    
class Multiscale_CNN2(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Multiscale_CNN2, self).__init__()
        self.wideconv = nn.Conv1d(in_channel, out_channel, kernel_size=32, stride=stride, padding=16, bias=False)
        self.conv1 = nn.Conv1d(out_channel, out_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=stride)
        self.conv4 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.sa = Self_Attention()
    def forward(self, input):
        x0 = self.wideconv(input) #(64,4,1024)
        x1 = self.conv1(x0) #(64,4,1024)
        x2_1 = self.conv1(x0) #(64,4,1024)
        x2_2 = self.conv2(x2_1) #(64,4,1024)
        x3_1 = self.conv1(x0) #(64,4,1024)
        x3_2 = self.conv2(x3_1) #(64,4,1024)
        x3_3 = self.conv3(x3_2) #(64,4,1024)
        x4_1 = self.maxpool(x0) #(64,4,1024)
        x4_2 = self.conv4(x4_1) #(64,4,1024)
        x = torch.cat([x1, x2_2, x3_3, x4_2], dim=1) #(64,16,1024)
        x = self.sa(x)
        return x
    
class Singlescale_CNN(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Singlescale_CNN, self).__init__()
        self.conv1 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, input):
        x1 = self.conv1(input) #(64,4,1024)
        x2 = self.conv1(input)
        x3 = self.conv1(input)
        x4 = self.conv1(input)
        x = torch.cat([x1, x2, x3, x4], dim=1) #(64,16,1024)
        return x

class Adaptive_Factor(nn.Module):
    def __init__(self, channel, gap_size):
        super(Adaptive_Factor, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),)
    def forward(self, x):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        out = out.unsqueeze(2)
        out = out * 0.005
        return out

class Shrinkage_Gap(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage_Gap, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),)
        self.adaptivefactor = Adaptive_Factor(channel, gap_size)
    def forward(self, x):
        x_raw = x
        x = torch.abs(x) #(64,16,1024)
        x = self.gap(x) #(64,16,1)
        emb1 = x
        x = torch.flatten(x, 1) #(64,16)
        out = self.fc(x) #(64,16)
        out = torch.mul(x, out) #(64,16)
        out = out.unsqueeze(2) #(64,16,1)
        h = self.adaptivefactor(x_raw) #(64,16,1)
        output = new_soft_thresholding(x_raw, out, h) #(64,16,1024)

        # x_raw = x
        # x = torch.abs(x)
        # x_abs = x
        # x = self.gap(x)
        # x = torch.flatten(x, 1)
        # average = x
        # x = self.fc(x)
        # x = torch.mul(average, x)
        # x = x.unsqueeze(2)
        # sub = x_abs - x
        # zeros = sub - sub
        # n_sub = torch.max(sub, zeros)
        # output = torch.mul(torch.sign(x_raw), n_sub)

        return output
    
class Shrinkage_Map(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage_Map, self).__init__()
        self.map = nn.MaxPool1d(kernel_size=64, stride=64)
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),)
        self.adaptivefactor = Adaptive_Factor(channel, gap_size)
    def forward(self, x):
        x_raw = x
        x = torch.abs(x) #(64,16,1024)
        x = self.map(x) #(64,16,16)
        x = self.gap(x) #(64,16,1)
        emb2 = x
        x = torch.flatten(x, 1) #(64,16)
        out = self.fc(x) #(64,16)
        out = torch.mul(x, out) #(64,16)
        out = out.unsqueeze(2) #(64,16,1)
        h = self.adaptivefactor(x_raw) #(64,16,1)
        output = new_soft_thresholding(x_raw, out, h) #(64,16,1024)

        # x_raw = x
        # x = torch.abs(x)
        # x_abs = x
        # x = self.gap(x)
        # x = torch.flatten(x, 1)
        # average = x
        # x = self.fc(x)
        # x = torch.mul(average, x)
        # x = x.unsqueeze(2)
        # sub = x_abs - x
        # zeros = sub - sub
        # n_sub = torch.max(sub, zeros)
        # output = torch.mul(torch.sign(x_raw), n_sub)

        return output

class AMARSN(nn.Module):
    def __init__(self, layers=[1, 1, 1, 1, 1], num_classes=7):    #layers=[1, 1, 1, 1]

        self.inplanes3 = 4
        super(AMARSN, self).__init__()
        self.upchannel_conv = nn.Conv1d(1, 4, kernel_size=1, stride=1, bias=False)
        self.downsample_conv = nn.Conv1d(16, 4, kernel_size=1, stride=1, bias=False)
        self.layer3x3_1 = self._make_layer(BasicBlock, 4, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer(BasicBlock, 4, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer(BasicBlock, 4, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer(BasicBlock, 4, layers[3], stride=2)
        self.layer3x3_5 = self._make_layer(BasicBlock, 4, layers[4], stride=2)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Linear(16, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.inplanes3, planes, stride))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x0):
        x = self.upchannel_conv(x0) #(64,4,2048)
        # x = self.drop1(x)
        x = self.layer3x3_1(x) #(64,16,1024)
        x = self.downsample_conv(x) #(64,4,1024)
        x = self.layer3x3_2(x) #(64,16,512)
        x = self.downsample_conv(x) #(64,4,512)
        # x = self.drop1(x)
        x = self.layer3x3_3(x) #(64,16,256)
        x = self.downsample_conv(x) #(64,4,256)
        x = self.layer3x3_4(x) #(64,16,128)
        x = self.downsample_conv(x) #(64,4,128)
        # x = self.drop1(x)
        x = self.layer3x3_5(x) #(64,16,64)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.squeeze()
        # x = self.drop2(x)
        out1 = self.fc(x)
        out2 = self.softmax(out1)
        # return emb1[2]
        return out1, out2
        # return out1
        # return torch.softmax(out1, dim=1)