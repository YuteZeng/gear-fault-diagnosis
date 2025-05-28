from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import *
import numpy as np
import torch
import itertools
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from Comparitive_Models.AMARSN import * 


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def wgn(x,snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    return noise

def laplace(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.laplace(0, 1, len(x)) * np.sqrt(Pn)
    return noise

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p,t] += 1
    return conf_matrix

batch_size = 128    #批大小为64
num_epochs = 450  #训练最大epoch为100
K = 2048            #每个样本长度为2048
flag = 0            #用于后续给样本打标签的变量
data_train_label = []   #首先定义一个空列表，用于接收训练样本标签
data_test_label = []
data_train = []         #首先定义一个空列表，用于接收训练样本
data_test = []

#读取TXT文件的数据
root_path = r'work1/HITdataset/Dataset/'  #文件路径（这里选择了实验室1027的MFS轴承数据集）
dirs = os.listdir(root_path)
for one_dir in dirs:
    full_dir = root_path + '/' + one_dir
    row_data = np.loadtxt(full_dir)     #读取数据
    row_data_0 = row_data[0:(len(row_data)//K)*K]   #根据K值，即2048，将整个数据化整，变成2048的整数倍；（这是为了某些数据不等长做的准备）
    row_data_1 = row_data_0.reshape(len(row_data_0)//K, K).tolist()    #将一维数据变成 N*2048 的矩阵
    for i in range(len(row_data_0)//K):
        if (i < (int(len(row_data_0)//K) * 0.8)):       #训练样本占总样本的0.8
            data_train.append(row_data_1[i])            #给训练样本打标签
            data_train_label.append(flag)
        else:
            data_test.append(row_data_1[i])             #测试样本占总样本的0.2
            data_test_label.append(flag)                #给测试样本打标签
    flag += 1


#将标记好标签的测试样本变成一维列表，随后变成数组类型
data_test_1D = sum(data_test, [])
data_test_1D = np.array(data_test_1D)
wgn_noise_test = wgn(data_test_1D, +2)
lap_noise_test = laplace(data_test_1D, +2)
data_test_noise = data_test_1D + wgn_noise_test + lap_noise_test
# data_test_noise = data_test_1D + wgn_noise_test 
# data_test_noise = data_test_1D + lap_noise_test
data_test = data_test_noise.reshape((320, 2048))

#测试标签从列表转数组
data_test_label = np.array(data_test_label)

#整合测试样本和测试标签，集成到Dataloader中
num_test_instances = len(data_test)
data_test = torch.from_numpy(data_test).type(torch.FloatTensor)
data_test_label = torch.from_numpy(data_test_label).type(torch.LongTensor)
data_test = data_test.view(num_test_instances, 1, -1)
data_test_label = data_test_label.view(num_test_instances, 1)
test_dataset = TensorDataset(data_test, data_test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

MyNetwork = torch.load('work1/HITdataset/AMARSN_DiagnosisResults/+6dB/Fault_diagnosis_exp/Mixed/Training accuracy100.0Validation_accuracy83.4375.pkl')
MyNetwork = MyNetwork.cuda()
MyNetwork.eval()
conf_matrix = torch.zeros(4, 4)
for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())
        predict_label, _ = MyNetwork(samplesV)
        predict_label = list(predict_label)
        predict_label = torch.tensor([item.cpu().detach().numpy() for item in predict_label]).cuda()
        conf_matrix = confusion_matrix(predict_label, labelsV, conf_matrix)
conf_matrix = np.array(conf_matrix.cpu())
per_kinds = conf_matrix.sum(axis=0)
corrects = conf_matrix.diagonal(offset=0)

print("混淆矩阵总元素个数： {0},测试集总个数：{1}".format(int(np.sum(conf_matrix)), 320))
print("每种故障总个数：", per_kinds)
print("每种故障分类正确总个数：", corrects)
print("每种故障的分类准确率为：{0}".format([rate*100 for rate in corrects/per_kinds]))

classes = ['C1', 'C2', 'C3', 'C4']

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, format(cm[i, j], '.2f'),
                horizontalalignment="center", verticalalignment="center", fontsize=18,
                color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, format(int(cm[i, j]), 'd'),
                     horizontalalignment="center", verticalalignment="center", fontsize=18,
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=24)
    plt.xlabel('Predicted label', fontsize=24)
    plt.savefig('work1/HITdataset/Comparitive_results/DRSN/0dB/confusion_matrix_AMARSN.png', format='png', bbox_inches='tight', dpi=800)
    plt.show()


plot_confusion_matrix(conf_matrix, classes=classes, normalize=False, title='Normalized Confusion Matrix')