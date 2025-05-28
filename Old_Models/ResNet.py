import torch
import torch.nn as nn


def conv1x3(in_planes, out_planes, stride=1):
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock1x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock1x3, self).__init__()
        #residual function
        self.conv1 = conv1x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, input_channel=1, layers=[1, 2, 1, 2, 1, 2], num_classes=7):

        self.inplanes3 = 4

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.drop = nn.Dropout(p=0.2)
        self.layer1x3_1 = self._make_layer(BasicBlock1x3, 4, layers[0], stride=2)
        self.layer1x3_2 = self._make_layer(BasicBlock1x3, 4, layers[1], stride=1)
        self.layer1x3_3 = self._make_layer(BasicBlock1x3, 8, layers[2], stride=2)
        self.layer1x3_4 = self._make_layer(BasicBlock1x3, 8, layers[3], stride=1)
        self.layer1x3_5 = self._make_layer(BasicBlock1x3, 16, layers[4], stride=2)
        self.layer1x3_6 = self._make_layer(BasicBlock1x3, 16, layers[5], stride=1)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(16, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def _make_layer(self, block, planes, blocks, stride):
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
        x = self.conv1(x0)
        x = self.drop(x)
        x = self.layer1x3_1(x)
        x = self.layer1x3_2(x)
        x = self.layer1x3_3(x)
        x = self.layer1x3_4(x)
        x = self.layer1x3_5(x)
        x = self.layer1x3_6(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        out = torch.cat([x], dim=1)
        out = out.squeeze()
        out1 = self.fc(out)
        out2 = self.softmax(out)
        return out1, out2
