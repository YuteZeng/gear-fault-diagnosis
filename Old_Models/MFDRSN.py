import torch
import torch.nn as nn
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride,
                     padding=2, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=17, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=33, stride=stride,
                     padding=1, bias=False)


def new_soft_thresholding(x, t):
    return x + (1/2)*(((x-t)**2+0.01)**0.5 - ((x+t)**2+0.01)**0.5)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        #residual function
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=9, stride=1, padding=5, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shrinkage = Shrinkage(planes, gap_size=(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        def FFT_CHANGE(x):
            x = np.abs(fft(x))
            y = x / np.max(np.abs(x))
            return y

        emb1 = self.conv1.weight.detach()
        print(emb1[3])
        emb1 = [0.9052, -0.2457,  1.4180,  0.1419, -0.9253, -1.0627,  0.2896,  0.4949,
          1.2205]
        emb1 = np.array(emb1)
        emb2 = FFT_CHANGE(emb1)
        emb1 = emb1.squeeze()
        emb2 = emb2.squeeze()
        x1 = range(9)
        plt.subplot(2, 1, 1)
        plt.plot(x1, emb1)
        x2 = range(9)
        plt.subplot(2, 1, 2)
        plt.plot(x2, emb2)
        plt.show()


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.shrinkage(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shrinkage = Shrinkage(planes, gap_size=(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.shrinkage(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shrinkage = Shrinkage(planes, gap_size=(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.shrinkage(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x
        # return new_soft_thresholding(x_raw, x)


class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=7):    #layers=[1, 1, 1, 1]

        self.inplanes3 = 4
        self.inplanes5 = 4
        self.inplanes7 = 4

        super(MSResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 4, kernel_size=64, stride=2, padding=31, bias=False)
        self.drop = nn.Dropout(p=0.2)
        # self.bn1 = nn.BatchNorm1d(4)
        # self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 4, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 8, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 16, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 32, layers[3], stride=2)
        # self.layer3x3_5 = self._make_layer3(BasicBlock3x3, 64, layers[4], stride=2)
        # self.layer3x3_6 = self._make_layer3(BasicBlock3x3, 128, layers[5], stride=2)
        # self.layer3x3_7 = self._make_layer3(BasicBlock3x3, 256, layers[6], stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))

        # maxplooing kernel size: 16, 11, 6

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 4, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 8, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 16, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 32, layers[3], stride=2)
        # self.layer5x5_5 = self._make_layer5(BasicBlock5x5, 64, layers[4], stride=2)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 4, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 8, layers[1], stride=2)
        # self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 16, layers[2], stride=2)
        #self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))

        self.fc = nn.Linear(32+16+8, num_classes)


        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        # emb1 = self.conv1.weight.detach()
        # print(emb1)
        x0 = self.conv1(x0)
        x0 = self.drop(x0)
        # x0 = self.bn1(x0)
        # x0 = self.relu(x0)
        # print(x0.shape)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        x = self.layer3x3_4(x)
        # x = self.layer3x3_5(x)
        # x = self.layer3x3_6(x)
        # x = self.layer3x3_7(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        # y = self.layer5x5_4(y)
        # y = self.layer5x5_5(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.avg_pool(y)

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        # z = self.layer7x7_3(z)
        #z = self.layer7x7_4(z)
        z = self.bn4(z)
        z = self.relu(z)
        z = self.avg_pool(z)

        #out = torch.cat([x], dim=1)
        out = torch.cat([x, y, z], dim=1)
        out = out.squeeze()
        out1 = self.fc(out)


        # return emb1[3], out1
        return out1
        # return torch.softmax(out1, dim=1)
