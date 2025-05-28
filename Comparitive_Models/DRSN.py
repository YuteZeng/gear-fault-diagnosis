import math
import torch
import torch.nn as nn
from torch import erf


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def new_soft_thresholding(x, t):
    return x + (1/2)*(((x-t)**2+0.001)**0.5 - ((x+t)**2+0.001)**0.5)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        #residual function
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shrinkage = Shrinkage(planes, gap_size=(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.shrinkage(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        #out = gelu(out)

        return out


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


class DRSN(nn.Module):
    def __init__(self, input_channel=1, layers=[1, 3, 1, 3, 1, 3], num_classes=4):    #layers=[1, 1, 1, 1]

        self.inplanes3 = 4
        super(DRSN, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 4, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.drop = nn.Dropout(p=0.2)
        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 4, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 4, layers[1], stride=1)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 8, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 8, layers[3], stride=1)
        self.layer3x3_5 = self._make_layer3(BasicBlock3x3, 16, layers[4], stride=2)
        self.layer3x3_6 = self._make_layer3(BasicBlock3x3, 16, layers[5], stride=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(16, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride):
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

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.drop(x0)
        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        x = self.layer3x3_4(x)
        x = self.layer3x3_5(x)
        x = self.layer3x3_6(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        out = torch.cat([x], dim=1)
        out = out.squeeze()
        out1 = self.fc(out)
        emb = out1
        out2 = self.softmax(out1)

        # return  emb, out1, out2
        return  out1, out2
        # return torch.softmax(out1, dim=1)