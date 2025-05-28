import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import torch



with open('work1/New_DRSN_DiagnosisResults/-6dB/模型验证精度和损失/validaacc.csv', encoding='utf-8') as csvfile:
    csv_reader_1_ACC=csv.reader(csvfile)
    acc1 =[row[0] for row in csv_reader_1_ACC]
with open('work1/New_DRSN_DiagnosisResults/-6dB/模型验证精度和损失/validaloss.csv', encoding='utf-8') as csvfile:
    csv_reader_1_loss=csv.reader(csvfile)
    loss1 =[row[0] for row in csv_reader_1_loss]


with open('work1/New_DRSN_DiagnosisResults/0dB/模型验证精度和损失/validaacc.csv') as csvfile:
    csv_reader_2_ACC=csv.reader(csvfile)
    acc2 =[row[0] for row in csv_reader_2_ACC]
    acc2 = np.array(acc2)
    acc2 = torch.form_numpy(acc2)
    print(acc2.shape)
    acc2 = np.reshape(1,450)
    print(acc2)
with open('work1/New_DRSN_DiagnosisResults/0dB/模型验证精度和损失/validaloss.csv') as csvfile:
    csv_reader_2_loss=csv.reader(csvfile)
    loss2 =[row[0] for row in csv_reader_2_loss]
    loss2 = np.array(loss2)

# with open('work1/New_DRSN_DiagnosisResults/+6dB/模型验证精度和损失/validaacc.csv', encoding='utf-8') as csvfile:
#     csv_reader_3_ACC=csv.reader(csvfile)
#     acc3 =[row[0] for row in csv_reader_3_ACC]
# with open('work1/New_DRSN_DiagnosisResults/+6dB/模型验证精度和损失/validaloss.csv', encoding='utf-8') as csvfile:
#     csv_reader_3_loss=csv.reader(csvfile)
#     loss3 =[row[0] for row in csv_reader_3_loss]




x = range(450)
# 画出生成数据图
# plt.plot(x, acc1, marker='.', label="dloss")
plt.plot(x, acc2, marker='.', label='gloss')
# plt.plot(x, acc3, marker='.', label='gloss')
plt.legend(loc='upper right')
plt.savefig('work1/New_DRSN_DiagnosisResults/0dB/模型验证精度和损失/ACC.png', format='png', bbox_inches='tight', dpi=800)