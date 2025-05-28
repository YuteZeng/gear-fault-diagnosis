import datetime
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from Comparitive_Models.AMARSN import * 
from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mclors

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

batch_size = 400    #批大小为64
num_epochs = 450  #训练最大epoch为100
K = 2048            #每个样本长度为2048
flag = 0            #用于后续给样本打标签的变量
data_train_label = []   #首先定义一个空列表，用于接收训练样本标签
data_test_label = []
data_train = []         #首先定义一个空列表，用于接收训练样本
data_test = []

#读取TXT文件的数据
root_path = r'work1/Dataset_three/'  #文件路径（这里选择了实验室1027的MFS轴承数据集）
dirs = os.listdir(root_path)
for one_dir in dirs:
    full_dir = root_path + '/' + one_dir
    row_data = np.loadtxt(full_dir)     #读取数据
    row_data_0 = row_data[0:(len(row_data)//K)*K]   #根据K值，即2048，将整个数据化整，变成2048的整数倍；（这是为了某些数据不等长做的准备）
    row_data_1 = row_data_0.reshape(len(row_data_0)//K, K).tolist()    #将一维数据变成 N*2048 的矩阵
    for i in range(len(row_data_0)//K):
        if (i < (int(len(row_data_0)//K) * 0.2)):       #训练样本占总样本的0.8
            data_train.append(row_data_1[i])            #给训练样本打标签
            data_train_label.append(flag)
        else:
            data_test.append(row_data_1[i])             #测试样本占总样本的0.2
            data_test_label.append(flag)                #给测试样本打标签
    flag += 1


#将标记好标签的测试样本变成一维列表，随后变成数组类型
data_test_1D = sum(data_test, [])
data_test_1D = np.array(data_test_1D)
wgn_noise_test = wgn(data_test_1D, 0)
lap_noise_test = laplace(data_test_1D, 0)
data_test_noise = data_test_1D
# data_test_noise = data_test_1D + wgn_noise_test 
# data_test_noise = data_test_1D + lap_noise_test
data_test = data_test_noise.reshape((400, 2048))

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

MyNetwork = torch.load('work1/结果2/Training accuracy100.0Validation_accuracy100.0.pkl')
MyNetwork = MyNetwork.cuda()
MyNetwork.eval()
embs = []
target = []
correct_test = 0
starttime_set()
for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())
        emb, predict_label, _ = MyNetwork(samplesV)     #模型输出
        embs.append(emb.data.cpu().numpy())
        target.append(labels.data.cpu().numpy())
        prediction = predict_label.data.max(1)[1]
        correct_test += prediction.eq(labelsV.data.long()).sum()

# x0 = x0[0].cpu()
# x0 = x0.squeeze()
# b = a[4].cpu()
# y1 = range(2048)
# y2 = range(64)
# # plt.plot(y1,x0)
# plt.plot(y2,b)
# plt.savefig('work1/New_DRSN_DiagnosisResults/+6dB/故障诊断实验/混合噪声/信号经过模块特征7.png', format='png', bbox_inches='tight', dpi=800)
# exit()

embs = np.concatenate(embs)
embs = np.concatenate(embs, axis=0)
embs = embs.reshape(400, 4)
target = np.concatenate(target)
tsne = TSNE(n_components=2, learning_rate=200, metric='cosine', n_jobs=-1)
tsne.fit_transform(embs)
outs_2d = np.array(tsne.embedding_)
css4 = list(mclors.CSS4_COLORS.keys())
css4 = ['green', 'yellow', 'blue', 'red']
css5 = ['o', 'o', 'o', 'o']
plt.figure(figsize=(10,6))
for lbi in range(4):
    temp = outs_2d[target==lbi]
    plt.plot(temp[:, 0], temp[:, 1], css5[lbi], color=css4[lbi], label='C'+str(lbi+1), markersize=10.0)
endtime_set()
plt.xlabel("Dimension1", fontsize=24)
plt.ylabel("Dimension2", fontsize=24)
plt.xlim(-40,30)
plt.ylim(-30,30)
plt.legend(fontsize=14)
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
print("Test accuracy:", (100 * float(correct_test) / num_test_instances))
plt.savefig('work1/结果2/TSNE.png', format='png', bbox_inches='tight', dpi=800)
plt.show()

