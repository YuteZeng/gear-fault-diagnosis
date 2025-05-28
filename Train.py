import os
import datetime
from torch.utils.data import TensorDataset, DataLoader
from Comparitive_Models.AMARSN import * 
from torch.autograd import Variable
from tqdm import tqdm
import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile, clever_format
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#定义训练开始时间函数，用于后续计算训练开始的时间
def starttime_set():
    global starttime
    starttime = datetime.datetime.now()
    print('Begin Time:', starttime)
    return starttime

#定义训练结束时间函数，用于后续计算训练结束的时间
def endtime_set():
    global endtime
    endtime = datetime.datetime.now()
    print('End Time:', endtime)
    return endtime

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


batch_size = 128   #批大小为64
num_epochs = 200 #训练最大epoch为100
K = 2048            #每个样本长度为2048
flag = 0            #用于后续给样本打标签的变量
data_train_label = []   #首先定义一个空列表，用于接收训练样本标签
data_test_label = []
data_train = []         #首先定义一个空列表，用于接收训练样本
data_test = []

#读取TXT文件的数据
root_path = r'MFSdataset/Dataset/'  #文件路径（这里选择了实验室1027的MFS轴承数据集）
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



#将标记好标签的训练样本变成一维列表，随后变成数组类型
data_train_1D = sum(data_train, [])
data_train_1D = np.array(data_train_1D)
wgn_noise_train = wgn(data_train_1D, +6)
lap_noise_train = laplace(data_train_1D, +6)
data_train_noise = data_train_1D
# data_train_noise = data_train_1D + wgn_noise_train
# data_train_noise = data_train_1D + lap_noise_train
data_train = data_train_noise.reshape((700, 2048))

#将标记好标签的测试样本变成一维列表，随后变成数组类型
data_test_1D = sum(data_test, [])
data_test_1D = np.array(data_test_1D)
wgn_noise_test = wgn(data_test_1D, +6)
lap_noise_test = laplace(data_test_1D, +6)
data_test_noise = data_test_1D
# data_test_noise = data_test_1D + wgn_noise_test
# data_test_noise = data_test_1D + lap_noise_test
data_test = data_test_noise.reshape((175, 2048))


#训练标签从列表转数组
data_train_label = np.array(data_train_label)
#测试标签从列表转数组
data_test_label = np.array(data_test_label)

#整合训练样本和训练标签，集成到Dataloader中
num_train_instances = len(data_train)
data_train = torch.from_numpy(data_train).type(torch.FloatTensor)
data_train_label = torch.from_numpy(data_train_label).type(torch.LongTensor)
data_train = data_train.view(num_train_instances, 1, -1)
data_train_label = data_train_label.view(num_train_instances, 1)
train_dataset = TensorDataset(data_train, data_train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#整合测试样本和测试标签，集成到Dataloader中
num_test_instances = len(data_test)
data_test = torch.from_numpy(data_test).type(torch.FloatTensor)
data_test_label = torch.from_numpy(data_test_label).type(torch.LongTensor)
data_test = data_test.view(num_test_instances, 1, -1)
data_test_label = data_test_label.view(num_test_instances, 1)
test_dataset = TensorDataset(data_test, data_test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#定义网络的输入，层数和类别数
MyNetwork = AMARSN(layers=[1, 1, 1, 1, 1], num_classes=7)
MyNetwork = MyNetwork.cuda()

#计算整个网络的参数数量
# input_shape = (1, 2048)
# summary(MyNetwork , input_shape)
# input_tensor = torch.randn(1, *input_shape).cuda()
# flops, params = profile(MyNetwork, inputs=(input_tensor,))
# flops, params = clever_format([flops, params], "%.3f")
# print("FLOPs: %s" %(flops))
# print("params: %s" %(params))


#损失函数，优化器
criterion = nn.NLLLoss().cuda()
optimizer = torch.optim.Adam(MyNetwork.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1) #每40个epoch，学习率*0.1

#定义空列表，用于填装loss
valida_loss = np.zeros([num_epochs, 1])
valida_acc = np.zeros([num_epochs, 1])
valida_loss_plot = []
valida_acc_plot = []


#训练开始
starttime_set()
for epoch in range(num_epochs):
    print('Epoch:', epoch)
    MyNetwork.train()
    scheduler.step()
    loss_x = 0
    for (samples, labels) in tqdm(train_data_loader):       #进度加载条
        samplesV = Variable(samples.cuda())     #样本变成变量形式，并且转GPU
        labels = labels.squeeze()               #标签压缩，减少冗余维度，用于计算损失
        labelsV = Variable(labels.cuda())       #标签变成变量形式，并且转GPU
        optimizer.zero_grad()
        out1, predict_label = MyNetwork(samplesV)     #模型输出
        loss = criterion(predict_label, labelsV)    #交叉熵损失，这里pytorch的交叉熵自带softmax
        loss_x += loss.item()
        loss.backward()
        optimizer.step()
    MyNetwork.eval()
    loss_x_train = 0
    correct_train = 0
    for i, (samples, labels) in enumerate(train_data_loader):   #利用训练样本正式训练模型
        with torch.no_grad():
            samplesV = Variable(samples.cuda())     #样本变成变量形式，并且转GPU
            labels = labels.squeeze()               #标签压缩，减少冗余维度，用于计算损失
            labelsV = Variable(labels.cuda())       #标签变成变量形式，并且转GPU
            predict_p1, predict_p2 = MyNetwork(samplesV)     #模型输出
            predict_label_train = predict_p2.data.max(1)[1]  #取预测标签的最大值，这里和softmax性质一样
            predict_p1 = predict_p1.data.max(1)[1]
            correct_train += predict_p1.eq(labelsV.data.long()).sum()   #判断与标签接近的值并求和用于后续计算精度
            loss = criterion(predict_p2, labelsV)
            loss_x_train += loss.item()
    print("Training_loss:", (loss_x_train/700))
    print("Training_accuracy:", (100*float(correct_train)/700))
    print('Train_macro_precision:',100*precision_score(labelsV.cpu(), predict_label_train.cpu(), average='macro'), 
        'Train_macro_recall:', 100*recall_score(labelsV.cpu(), predict_label_train.cpu(), average='macro'), 
        'Train_macro_f1_score:', 100*f1_score(labelsV.cpu(), predict_label_train.cpu(), average='macro'))
    loss_x2_valid = 0
    correct_valid = 0
    for i, (samples, labels) in enumerate(test_data_loader):   #验证样本验证模型
        with torch.no_grad():
            samplesV = Variable(samples.cuda())     #样本变成变量形式，并且转GPU
            labels = labels.squeeze()               #标签压缩，减少冗余维度，用于计算损失
            labelsV = Variable(labels.cuda())       #标签变成变量形式，并且转GPU
            predict_p3, predict_p4 = MyNetwork(samplesV)     #模型输出
            prediction_label_valid = predict_p4.data.max(1)[1]  #取预测标签的最大值，这里和softmax性质一样
            predict_p3 = predict_p3.data.max(1)[1]
            correct_valid += predict_p3.eq(labelsV.data.long()).sum()   #判断与标签接近的值并求和用于后续计算精度
            loss = criterion(predict_p4, labelsV)
            loss_x2_valid += loss.item()
    valida_loss[epoch] = (loss_x2_valid/175)
    valida_loss_plot.append(valida_loss[epoch])
    valida_acc[epoch] = (100*float(correct_valid)/175)
    valida_acc_plot.append(valida_acc[epoch])
    print("Validation_loss:", (loss_x2_valid/175))
    print("Validation_accuracy:", (100*float(correct_valid)/175))
    print('Valid_macro_precision:', 100*precision_score(labelsV.cpu(), prediction_label_valid.cpu(), average='macro'), 
        'Valid_macro_recall:', 100*recall_score(labelsV.cpu(), prediction_label_valid.cpu(), average='macro'), 
        'Valid_macro_f1_score:', 100*f1_score(labelsV.cpu(), prediction_label_valid.cpu(), average='macro'))


Training_accuracy = str(100*float(correct_train)/1280)
Validation_accuracy = str(100*float(correct_valid)/320)

torch.save(MyNetwork, 'random/20250527/Training accuracy' + Training_accuracy + 
           'Validation_accuracy' + Validation_accuracy + '.pkl')
# torch.save(MyNetwork, 'work1/HITdataset/AMARSN_DiagnosisResults/+2dB/Fault_diagnosis_exp/Gussian/Training accuracy' + Training_accuracy + 
#            'Validation_accuracy' + Validation_accuracy + '.pkl')
# torch.save(MyNetwork, 'work1/HITdataset/AMARSN_DiagnosisResults/+2dB/Fault_diagnosis_exp/Laplace/Training accuracy' + Training_accuracy + 
#            'Validation_accuracy' + Validation_accuracy + '.pkl')

endtime_set()

# pd.DataFrame(valida_acc).to_csv('work1/HITdataset/Comparitive_results/AMARSN/0dB/validaacc-6.csv', index=False)
# pd.DataFrame(valida_loss).to_csv('work1/HITdataset/Comparitive_results/AMARSN/0dB/validaloss-6.csv', index=False)
#结束训练
