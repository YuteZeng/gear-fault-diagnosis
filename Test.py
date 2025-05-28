import os
import datetime
from torch.utils.data import TensorDataset, DataLoader
from Comparitive_Models.AMARSN import * 
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score
import warnings
import pandas as pd
import seaborn as sns
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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


batch_size = 175    #批大小为64
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

#将标记好标签的测试样本变成一维列表，随后变成数组类型
data_test_1D = sum(data_test, [])
data_test_1D = np.array(data_test_1D)
wgn_noise_test = wgn(data_test_1D, -6)
lap_noise_test = laplace(data_test_1D, 0)
# data_test_noise = data_test_1D + wgn_noise_test + lap_noise_test
data_test_noise = data_test_1D
# data_test_noise = data_test_1D + lap_noise_test
data_test = data_test_noise.reshape((175, 2048))


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

MyNetwork = torch.load('random/20250527/Training accuracy52.96875Validation_accuracy52.8125.pkl')
MyNetwork = MyNetwork.cuda()
MyNetwork.eval()
correct_test = 0

starttime_set()
for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())     #样本变成变量形式，并且转GPU
        labels = labels.squeeze()               #标签压缩，减少冗余维度，用于计算损失
        labelsV = Variable(labels.cuda())       #标签变成变量形式，并且转GPU
        predict_p3, predict_p4 = MyNetwork(samplesV)     #模型输出
        prediction_label_valid = predict_p4.data.max(1)[1]  #取预测标签的最大值，这里和softmax性质一样
        predict_p3 = predict_p3.data.max(1)[1]
        print("测试样本中的前十个样例标签如下：",predict_p3[:10])
        correct_test += predict_p3.eq(labelsV.data.long()).sum()   #判断与标签接近的值并求和用于后续计算精度

print("Testing_accuracy:", (100*float(correct_test)/175))
print('Test_macro_precision:',100*precision_score(labelsV.cpu(), prediction_label_valid.cpu(), average='macro'), 
        'Test_macro_recall:', 100*recall_score(labelsV.cpu(), prediction_label_valid.cpu(), average='macro'), 
        'Test_macro_f1_score:', 100*f1_score(labelsV.cpu(), prediction_label_valid.cpu(), average='macro'))
endtime_set()

# starttime_set()
# for i, (samples, labels) in enumerate(test_data_loader):
#     with torch.no_grad():
#         samplesV = Variable(samples.cuda())     #样本变成变量形式，并且转GPU
#         residual = MyNetwork(samplesV)     #模型输出

# residual = residual.reshape(64,16)
# residual = residual.cpu()
# embs = np.matrix(residual)
# f, ax = plt.subplots()
# sns.heatmap(data=embs, center=0, cmap='YlGnBu_r', xticklabels=True, yticklabels=True)
# ax.set_xlabel("Dimension", labelpad=20)
# ax.set_ylabel("Token", labelpad=20)
# plt.savefig('work1/New_DRSN_DiagnosisResults/0dB/消融研究/多尺度注意特征提取/AMARSN/自注意力权重分配1.png', format='png', bbox_inches='tight', dpi=800)
exit()

# k = range(16)
# a0 = emb1[1]
# a0 = a0.cpu()
# a1 = emb2[1]
# a1 = a1.cpu()
# plt.plot(k, a0, color="orange")
# plt.plot(k, a1, color="blue")
# plt.savefig('work1/New_DRSN_DiagnosisResults/0dB/消融研究/多尺度注意特征提取/AMARSN/全局局部特征.png', format='png', bbox_inches='tight', dpi=800)
# pd.DataFrame(a0).to_csv('work1/New_DRSN_DiagnosisResults/0dB/消融研究/多尺度注意特征提取/AMARSN/全局特征.csv',index=False)
# pd.DataFrame(a1).to_csv('work1/New_DRSN_DiagnosisResults/0dB/消融研究/多尺度注意特征提取/AMARSN/局部特征.csv',index=False)


endtime_set()



