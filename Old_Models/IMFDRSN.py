import torch
import torch.nn as nn


# 定义的1*9卷积
def conv1x9(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride,
                     padding=2, bias=False)

# 定义的1*17卷积
def conv1x17(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=17, stride=stride,
                     padding=1, bias=False)

# 定义的1*33卷积
def conv1x33(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=33, stride=stride,
                     padding=1, bias=False)

# 改进的软阈值函数
def new_soft_thresholding(x, t):
    return x + (1/2)*(((x-t)**2+0.003)**0.5 - ((x+t)**2+0.003)**0.5)

# 分支1中的单个block,这里的block就是IRSBU
class BasicBlock1x9(nn.Module):
    expansion = 1

    # 两个卷积，两个Bn，两个relu，一个改进的收缩模块，残差链接
    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock1x9, self).__init__()
        #residual function
        self.conv1 = conv1x9(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=9, stride=1, padding=5, bias=False)
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

        out += residual
        out = self.relu(out)

        return out

# 分支2中的单个block,这里的block就是IRSBU
class BasicBlock1x17(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock1x17, self).__init__()
        self.conv1 = conv1x17(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x17(planes, planes)
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

# 分支3中的单个block,这里的block就是IRSBU
class BasicBlock1x33(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock1x33, self).__init__()
        self.conv1 = conv1x33(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x33(planes, planes)
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

# 改进的收缩模块
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
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        #使用改进的软阈值来处理输入特征
        return new_soft_thresholding(x_raw, x)

# 主模型

# 主模型
class IMFDRSN(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=7):    #layers=[1, 1, 1, 1]

        self.inplanes3 = 4
        self.inplanes5 = 4
        self.inplanes7 = 4

        super(IMFDRSN, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 4, kernel_size=64, stride=2, padding=31, bias=False) #宽卷积层
        self.drop = nn.Dropout(p=0.2)

        #################################################################################   分支1具有4个block
        self.layer1x9_1 = self._make_layer1x9(BasicBlock1x9, 4, layers[0], stride=2)
        self.layer1x9_2 = self._make_layer1x9(BasicBlock1x9, 8, layers[1], stride=2)
        self.layer1x9_3 = self._make_layer1x9(BasicBlock1x9, 16, layers[2], stride=2)
        self.layer1x9_4 = self._make_layer1x9(BasicBlock1x9, 32, layers[3], stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        #################################################################################

        #################################################################################   分支2具有3个block
        self.layer1x17_1 = self._make_layer1x17(BasicBlock1x17, 4, layers[0], stride=2)
        self.layer1x17_2 = self._make_layer1x17(BasicBlock1x17, 8, layers[1], stride=2)
        self.layer1x17_3 = self._make_layer1x17(BasicBlock1x17, 16, layers[2], stride=2)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        #################################################################################

        #################################################################################   分支3具有两个block
        self.layer1x33_1 = self._make_layer1x33(BasicBlock1x33, 4, layers[0], stride=2)
        self.layer1x33_2 = self._make_layer1x33(BasicBlock1x33, 8, layers[1], stride=2)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        #################################################################################

        self.fc = nn.Linear(32+16+8, num_classes)

    # 定义分支1
    def _make_layer1x9(self, block, planes, blocks, stride=2):
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

    # 定义分支2
    def _make_layer1x17(self, block, planes, blocks, stride=2):
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

    # 定义分支3
    def _make_layer1x33(self, block, planes, blocks, stride=2):
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
        x0 = self.conv1(x0) #宽卷积层
        x0 = self.drop(x0)

        ######################### 分支1
        x = self.layer1x9_1(x0)
        x = self.layer1x9_2(x)
        x = self.layer1x9_3(x)
        x = self.layer1x9_4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        #########################

        ######################### 分支2
        y = self.layer1x17_1(x0)
        y = self.layer1x17_2(y)
        y = self.layer1x17_3(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.avg_pool(y)
        #########################

        ######################### 分支3
        z = self.layer1x33_1(x0)
        z = self.layer1x33_2(z)
        z = self.bn4(z)
        z = self.relu(z)
        z = self.avg_pool(z)
        #########################

        # 特征拼接
        out = torch.cat([x, y, z], dim=1)
        out = out.squeeze()
        # 全连接层输出
        out1 = self.fc(out)

        return out1
