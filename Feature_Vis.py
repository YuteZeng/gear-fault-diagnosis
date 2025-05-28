import os
import datetime
from torch.utils.data import TensorDataset, DataLoader
from Comparitive_Models.AMARSN import * 
import scipy.io as sio
import seaborn as sns
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score
import warnings
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

def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) -np.min(x))
    return x

batch_size = 128    #批大小为64
num_epochs = 360  #训练最大epoch为100
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
        if (i < (int(len(row_data_0)//K) * 0.1)):       #训练样本占总样本的0.8
            data_train.append(row_data_1[i])            #给训练样本打标签
            data_train_label.append(flag)
        else:
            data_test.append(row_data_1[i])             #测试样本占总样本的0.2
            data_test_label.append(flag)                #给测试样本打标签
    flag += 1



#将标记好标签的训练样本变成一维列表，随后变成数组类型
data_train_1 = np.array(data_train[0])
wgn_noise_train_1 = wgn(data_train_1,  -6)
lap_noise_train_1 = laplace(data_train_1, -6)
data_train_1_noise = data_train_1

data_train_2 = np.array(data_train[45])
wgn_noise_train_2 = wgn(data_train_2, -6)
lap_noise_train_2 = laplace(data_train_2, -6)
data_train_2_noise = data_train_2

data_train_3 = np.array(data_train[90])
wgn_noise_train_3 = wgn(data_train_3, -6)
lap_noise_train_3 = laplace(data_train_3, -6)
data_train_3_noise = data_train_3

data_train_4 = np.array(data_train[135])
wgn_noise_train_4 = wgn(data_train_4, -6)
lap_noise_train_4 = laplace(data_train_4, -6)
data_train_4_noise = data_train_4

# data_train_5 = np.array(data_train[40])
# wgn_noise_train_5 = wgn(data_train_5, -6)
# lap_noise_train_5 = laplace(data_train_5, -6)
# data_train_5_noise = data_train_5 + wgn_noise_train_5 + lap_noise_train_5

# data_train_6 = np.array(data_train[50])
# wgn_noise_train_6 = wgn(data_train_6, -6)
# lap_noise_train_6 = laplace(data_train_6, -6)
# data_train_6_noise = data_train_6 + wgn_noise_train_6 + lap_noise_train_6

# data_train_7 = np.array(data_train[60])
# wgn_noise_train_7 = wgn(data_train_7, -6)
# lap_noise_train_7 = laplace(data_train_7, -6)
# data_train_7_noise = data_train_7 + wgn_noise_train_7 + lap_noise_train_7

sns.kdeplot(data_train_1_noise, shade=True, color='red', label = 'C1')
sns.kdeplot(data_train_2_noise, shade=True, color='blue', label = 'C2')
sns.kdeplot(data_train_3_noise, shade=True, color='green', label = 'C3')
sns.kdeplot(data_train_4_noise, shade=True, color='yellow', label = 'C4')
# sns.kdeplot(data_train_5_noise, shade=True, color='orange', label = 'C5')
# sns.kdeplot(data_train_6_noise, shade=True, color='purple', label = 'C6')
# sns.kdeplot(data_train_7_noise, shade=True, color='lightgreen', label = 'C7')
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Density', fontsize=14)

# plt.hist(lap_noise_train, bins=30, density=True, alpha=0.5,
#          histtype='stepfilled', color='steelblue',
#          edgecolor='none')
# # 绘制正态分布曲线
# mu, sigma = lap_noise_train.mean(), lap_noise_train.std()
# x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
# plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2)),
#          linewidth=2, color='r')

plt.savefig('work1/HITdataset/AMARSN_DiagnosisResults/+6dB/Fault_chagne_with_noise/feature_visua_comp/No_mixed_noise.png', bbox_inches = 'tight', format='png', dpi=800)
