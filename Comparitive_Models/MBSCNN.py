import torch
import torch.nn as nn
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

def guassian_func(x):
    delta = 1
    return 1/(delta*np.sqrt(2*np.pi))*np.exp(-x*x/(2*delta*delta))

def gaussian_filtering(x):
    x = x.cpu()
    x = torch.flatten(x)
    w_j = np.arange(5)-2
    guassian_coef = [guassian_func(i) for i in w_j]
    x = np.convolve(x, guassian_coef, 'same')/sum(guassian_coef)
    x = x.reshape(-1, 1, 2048)
    x = torch.from_numpy(x)
    x = x.cuda().float()
    return x

def moving_average(x, w=5, padding = 'same'):
    x = x.cpu()
    x = torch.flatten(x)
    x = np.convolve(x, np.ones(w), padding) / w
    x = x.reshape(-1 ,1, 2048)
    x = torch.from_numpy(x)
    x = x.cuda().float()
    return x

class Multiscale_CNN(nn.Module):
    def __init__(self):
        super(Multiscale_CNN, self).__init__()
        self.moudle1 = MultiscaleCNN_Module1()
        self.moudle2 = MultiscaleCNN_Module2()
        self.moudle3 = MultiscaleCNN_Module3()
        self.moudle4 = MultiscaleCNN_Module4()
        self.moudle5 = MultiscaleCNN_Module5()
    def forward(self, input):
        x = self.moudle1(input) 
        x = self.moudle2(x)
        x = self.moudle3(x) 
        x = self.moudle4(x) 
        x = self.moudle5(x) 
        return x
class MultiscaleCNN_Module1(nn.Module):
    def __init__(self):
        super(MultiscaleCNN_Module1, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=4, bias=False)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x1 = self.conv(input)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.drop(x1)
        x2 = self.conv(input)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x3 = self.conv(input)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        x3 = self.drop(x3)
        x_all = torch.cat([x1, x2, x3], dim=1) 
        return x_all
class MultiscaleCNN_Module2(nn.Module):
    def __init__(self):
        super(MultiscaleCNN_Module2, self).__init__()
        self.conv = nn.Conv1d(in_channels=48, out_channels=32, kernel_size=5, stride=4, bias=False)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x1 = self.conv(input)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.drop(x1)
        x2 = self.conv(input)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x3 = self.conv(input)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        x3 = self.drop(x3)
        x_all = torch.cat([x1, x2, x3], dim=1) 
        return x_all
class MultiscaleCNN_Module3(nn.Module):
    def __init__(self):
        super(MultiscaleCNN_Module3, self).__init__()
        self.conv = nn.Conv1d(in_channels=96, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x1 = self.conv(input)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.drop(x1)
        x2 = self.conv(input)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x3 = self.conv(input)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        x3 = self.drop(x3)
        x_all = torch.cat([x1, x2, x3], dim=1) 
        return x_all
class MultiscaleCNN_Module4(nn.Module):
    def __init__(self):
        super(MultiscaleCNN_Module4, self).__init__()
        self.conv = nn.Conv1d(in_channels=192, out_channels=128, kernel_size=3, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x1 = self.conv(input)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.drop(x1)
        x2 = self.conv(input)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x3 = self.conv(input)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        x3 = self.drop(x3)
        x_all = torch.cat([x1, x2, x3], dim=1) #(64,16,1024)
        return x_all   
class MultiscaleCNN_Module5(nn.Module):
    def __init__(self):
        super(MultiscaleCNN_Module5, self).__init__()
        self.conv = nn.Conv1d(in_channels=384, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x1 = self.conv(input)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.drop(x1)
        x2 = self.conv(input)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x3 = self.conv(input)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        x3 = self.drop(x3)
        x_all = torch.cat([x1, x2, x3], dim=1) 
        return x_all

class Signlescale_CNN(nn.Module):
    def __init__(self):
        super(Signlescale_CNN, self).__init__()
        self.moudle1 = SignlescaleCNN_Module1()
        self.moudle2 = SignlescaleCNN_Module2()
        self.moudle3 = SignlescaleCNN_Module3()
        self.moudle4 = SignlescaleCNN_Module4()
        self.moudle5 = SignlescaleCNN_Module5()
    def forward(self, input):
        x = self.moudle1(input) 
        x = self.moudle2(x) 
        x = self.moudle3(x) 
        x = self.moudle4(x) 
        x = self.moudle5(x) 
        return x
class SignlescaleCNN_Module1(nn.Module):
    def __init__(self):
        super(SignlescaleCNN_Module1, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=4, bias=False)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x
class SignlescaleCNN_Module2(nn.Module):
    def __init__(self):
        super(SignlescaleCNN_Module2, self).__init__()
        self.conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=4, bias=False)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x
class SignlescaleCNN_Module3(nn.Module):
    def __init__(self):
        super(SignlescaleCNN_Module3, self).__init__()
        self.conv = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x
class SignlescaleCNN_Module4(nn.Module):
    def __init__(self):
        super(SignlescaleCNN_Module4, self).__init__()
        self.conv = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x  
class SignlescaleCNN_Module5(nn.Module):
    def __init__(self):
        super(SignlescaleCNN_Module5, self).__init__()
        self.conv = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x) 
        return x

class MBSCNN(nn.Module):
    def __init__(self, layers=[1, 1, 1, 1], num_classes=4):    #layers=[1, 1, 1, 1]
        self.inplanes3 = 1
        super(MBSCNN, self).__init__()
        self.Multibranch = Multiscale_CNN()
        self.Denosingbranch = Signlescale_CNN()
        self.Lowfreqbranch = Signlescale_CNN()
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc2 = nn.Linear(1280, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.gap = nn.AdaptiveAvgPool1d((1))
    def forward(self, input):
        x0_1 = input
        x0_2 = gaussian_filtering(input)
        x0_3 = moving_average(input)
        x1 = self.Multibranch(x0_1)
        x2 = self.Denosingbranch(x0_2)
        x3 = self.Lowfreqbranch(x0_3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.gap(x)
        out = x.squeeze()
        out1 = self.fc2(out)
        out2 = self.softmax(out1)
        return out1, out2

