import torch
import torch.nn as nn
from torch import erf
import matplotlib.pyplot as plt


class MLMod1(nn.Module):
    def __init__(self):
        super(MLMod1, self).__init__()
        self.baseconv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=32, padding=15, stride=2, bias=False)
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input): #(B,1,2048)
        x0 = self.baseconv(input)  #(B,64,1024)
        x0_split = torch.split(x0, 16, 1)
        x1 = self.conv1(x0_split[0]) #(B,16,1024)
        x2 = torch.cat([x0_split[1], x1], dim=1)
        x2 = self.conv2(x2) #(B,16,1024)
        x3 = torch.cat([x0_split[2], x2], dim=1)
        x3 = self.conv3(x3) #(B,16,1024)
        x4 = torch.cat([x0_split[3], x3], dim=1)
        x4 = self.conv4(x4) #(B,16,1024)
        x_all = torch.cat([x1, x2, x3, x4], dim=1)
        x_all = self.bn(x_all)
        x_all = self.relu(x_all) #(B,64,1024)
        return x_all
class RAMod1(nn.Module):
    def __init__(self):
        super(RAMod1, self).__init__()
        self.conv = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.gap = nn.AdaptiveAvgPool1d((1))
        self.bn = nn.BatchNorm1d(64)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input): #(B,64,1024)
        x1 = self.gap(input) #(B,64,1)
        x2 = self.conv(input)
        x_all = torch.matmul(x1, x2) #(B,64,1024)
        x_all = self.bn(x_all)
        x_all = self.sigmoid(x_all)
        output = torch.mul(x_all, input) + input + x_all #(B,64,1024)
        return output

class MLMod2(nn.Module):
    def __init__(self):
        super(MLMod2, self).__init__()
        self.baseconv = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, padding=7, stride=2, bias=False)
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input): #(B,64,1024)
        x0 = self.baseconv(input)  #(B,64,512)
        x0_split = torch.split(x0, 16, 1)
        x1 = self.conv1(x0_split[0]) #(B,16,512)
        x2 = torch.cat([x0_split[1], x1], dim=1)
        x2 = self.conv2(x2) #(B,16,512)
        x3 = torch.cat([x0_split[2], x2], dim=1)
        x3 = self.conv3(x3) #(B,16,512)
        x4 = torch.cat([x0_split[3], x3], dim=1)
        x4 = self.conv4(x4) #(B,16,512)
        x_all = torch.cat([x1, x2, x3, x4], dim=1)
        x_all = self.bn(x_all)
        x_all = self.relu(x_all) #(B,64,512)
        return x_all
class RAMod2(nn.Module):
    def __init__(self):
        super(RAMod2, self).__init__()
        self.conv = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.gap = nn.AdaptiveAvgPool1d((1))
        self.bn = nn.BatchNorm1d(64)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input): #(B,64,512)
        x1 = self.gap(input) #(B,64,1)
        x2 = self.conv(input)
        x_all = torch.matmul(x1, x2) #(B,64,512)
        x_all = self.bn(x_all)
        x_all = self.sigmoid(x_all)
        output = torch.mul(x_all, input) + input + x_all #(B,64,512)
        return output

class MLMod3(nn.Module):
    def __init__(self):
        super(MLMod3, self).__init__()
        self.baseconv = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, padding=7, stride=2, bias=False)
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input): #(B,64,512)
        x0 = self.baseconv(input)  #(B,128,256)
        x0_split = torch.split(x0, 32, 1)
        x1 = self.conv1(x0_split[0]) #(B,32,256)
        x2 = torch.cat([x0_split[1], x1], dim=1)
        x2 = self.conv2(x2) #(B,32,256)
        x3 = torch.cat([x0_split[2], x2], dim=1)
        x3 = self.conv3(x3) #(B,32,256)
        x4 = torch.cat([x0_split[3], x3], dim=1)
        x4 = self.conv4(x4) #(B,32,256)
        x_all = torch.cat([x1, x2, x3, x4], dim=1)
        x_all = self.bn(x_all)
        x_all = self.relu(x_all) #(B,128,256)
        return x_all
class RAMod3(nn.Module):
    def __init__(self):
        super(RAMod3, self).__init__()
        self.conv = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.gap = nn.AdaptiveAvgPool1d((1))
        self.bn = nn.BatchNorm1d(128)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input): #(B,128,256)
        x1 = self.gap(input) #(B,128,1)
        x2 = self.conv(input)
        x_all = torch.matmul(x1, x2) #(B,128,256)
        x_all = self.bn(x_all)
        x_all = self.sigmoid(x_all)
        output = torch.mul(x_all, input) + input + x_all #(B,128,256)
        return output

class MLMod4(nn.Module):
    def __init__(self):
        super(MLMod4, self).__init__()
        self.baseconv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, padding=3, stride=2, bias=False)
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input): #(B,128,256)
        x0 = self.baseconv(input)  #(B,128,128)
        x0_split = torch.split(x0, 32, 1)
        x1 = self.conv1(x0_split[0]) #(B,32,128)
        x2 = torch.cat([x0_split[1], x1], dim=1)
        x2 = self.conv2(x2) #(B,32,128)
        x3 = torch.cat([x0_split[2], x2], dim=1)
        x3 = self.conv3(x3) #(B,32,128)
        x4 = torch.cat([x0_split[3], x3], dim=1)
        x4 = self.conv4(x4) #(B,32,128)
        x_all = torch.cat([x1, x2, x3, x4], dim=1)
        x_all = self.bn(x_all)
        x_all = self.relu(x_all) #(B,128,128)
        return x_all
class RAMod4(nn.Module):
    def __init__(self):
        super(RAMod4, self).__init__()
        self.conv = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.gap = nn.AdaptiveAvgPool1d((1))
        self.bn = nn.BatchNorm1d(128)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input): #(B,128,128)
        x1 = self.gap(input) #(B,128,1)
        x2 = self.conv(input)
        x_all = torch.matmul(x1, x2) #(B,128,128)
        x_all = self.bn(x_all)
        x_all = self.sigmoid(x_all)
        output = torch.mul(x_all, input) + input + x_all #(B,128,128)
        return output

class MLMod5(nn.Module):
    def __init__(self):
        super(MLMod5, self).__init__()
        self.baseconv = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2, bias=False)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input): #(B,128,128)
        x0 = self.baseconv(input)  #(B,256,64)
        x0_split = torch.split(x0, 64, 1)
        x1 = self.conv1(x0_split[0]) #(B,64,64)
        x2 = torch.cat([x0_split[1], x1], dim=1)
        x2 = self.conv2(x2) #(B,64,64)
        x3 = torch.cat([x0_split[2], x2], dim=1)
        x3 = self.conv3(x3) #(B,64,64)
        x4 = torch.cat([x0_split[3], x3], dim=1)
        x4 = self.conv4(x4) #(B,64,64)
        x_all = torch.cat([x1, x2, x3, x4], dim=1)
        x_all = self.bn(x_all)
        x_all = self.relu(x_all) #(B,256,64)
        return x_all
class RAMod5(nn.Module):
    def __init__(self):
        super(RAMod5, self).__init__()
        self.conv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.gap = nn.AdaptiveAvgPool1d((1))
        self.bn = nn.BatchNorm1d(256)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input): #(B,256,64)
        x1 = self.gap(input) #(B,256,1)
        x2 = self.conv(input)
        x_all = torch.matmul(x1, x2) #(B,256,64)
        x_all = self.bn(x_all)
        x_all = self.sigmoid(x_all)
        output = torch.mul(x_all, input) + input + x_all #(B,256,64)
        return output


class MRACNN(nn.Module):
    def __init__(self, layers=[1, 1, 1, 1, 1], num_classes=4):    #layers=[1, 1, 1, 1]
        super(MRACNN, self).__init__()
        self.MLMod1 = MLMod1()
        self.RAMod1 = RAMod1()
        self.MLMod2 = MLMod2()
        self.RAMod2 = RAMod2()
        self.MLMod3 = MLMod3()
        self.RAMod3 = RAMod3()
        self.MLMod4 = MLMod4()
        self.RAMod4 = RAMod4()
        self.MLMod5 = MLMod5()
        self.RAMod5 = RAMod5()
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(256, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x = self.MLMod1(input) 
        x = self.RAMod1(x) 
        x = self.MLMod2(x) 
        x = self.RAMod2(x) 
        x = self.MLMod3(x) 
        x = self.RAMod3(x) 
        x = self.MLMod4(x) 
        x = self.RAMod4(x) 
        x = self.MLMod5(x) 
        x = self.RAMod5(x) 
        x = self.avg_pool(x)
        out = x.squeeze()
        out1 = self.fc(out)
        out2 = self.softmax(out1)

        return out1, out2
