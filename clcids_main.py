import os
import random
import torch
import pandas as pd
from torch import nn
import numpy as np
from torch.optim import Adam
from utils.trainer import train
# from model.resnet import DeepNet
from model.denseNet import DeepNet
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 该项目中某些必要的参数
seed = 5
batch_size = 1024
epoch_num = 100
lr = 1e-3
output_path = 'output'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_names = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
              'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
              'Friday-WorkingHours-Morning.pcap_ISCX.csv',
              'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
              'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
              'Tuesday-WorkingHours.pcap_ISCX.csv',
              'Wednesday-workingHours.pcap_ISCX.csv'
              ]
label2num = [
    {'BENIGN': 0, 'DDoS': 1},
    {'BENIGN': 0, 'PortScan': 1},
    {'BENIGN': 0, 'Bot': 1},
    {'BENIGN': 0, 'Infiltration': 1},
    {'BENIGN': 0, 'Web Attack � Brute Force': 1, 'Web Attack � Sql Injection': 2,
     'Web Attack � XSS': 3},
    {'BENIGN': 0, 'FTP-Patator': 1, 'SSH-Patator': 2},
    {'BENIGN': 0, 'DoS GoldenEye': 1, 'DoS Hulk': 2, 'DoS Slowhttptest': 3, 'DoS slowloris': 4,
     'Heartbleed': 5}
]

num_classes = [2, 2, 2, 2, 4, 3, 6]

file_name_index = 5
data_path = 'F:\data\CIC-IDS-2017\MachineLearningCSV'
file_name = file_names[file_name_index]
num_class = num_classes[file_name_index]
data = pd.read_csv(os.path.join(data_path, file_name))
data['Label'] = data[' Label'].map(label2num[file_name_index])


# 下列特征在所有的样本中，值均为0，无用，所以删除
drop_column_list = [
    ' Label', ' Bwd PSH Flags', ' Bwd URG Flags', 'Fwd Avg Bytes/Bulk',
    ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',  ' Bwd Avg Bytes/Bulk',
    ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', ' Fwd URG Flags',
    ' CWE Flag Count'
]
for column_name in drop_column_list:
    data.drop(column_name, axis=1, inplace=True)

# 删除掉具有空值的样本
data.dropna(inplace=True)
# 'Flow Bytes/s'和' Flow Packets/s'特征中部分样本的值为无限大，在深度学习中，会发生梯度爆炸，无法参与运算,删除掉具有inf的样本
data = data[data['Flow Bytes/s'] != float('inf')]
data = data[data[' Flow Packets/s'] != float('inf')]
print(len(data))


def main():
    data_feature = data.drop('Label', axis=1, inplace=False)
    # 归一化
    data_feature = (data_feature - data_feature.min())/(data_feature.max()-data_feature.min())
    # 标准化
    #
    data_feature = data_feature.values
    labels = data['Label'].values.reshape(-1)

    train_input, test_input, train_label, test_label = train_test_split(data_feature, labels, test_size=0.2, random_state=seed)

    model = DeepNet(input_size=68, num_class=num_class, dropout=0.5)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criteria = nn.CrossEntropyLoss()

    train_set = TensorDataset(torch.tensor(train_input, dtype=torch.float),
                              torch.tensor(train_label, dtype=torch.long))
    train_iter = DataLoader(train_set, batch_size, shuffle=True)

    dev_set = TensorDataset(torch.tensor(test_input, dtype=torch.float),
                            torch.tensor(test_label, dtype=torch.long))
    dev_iter = DataLoader(dev_set, batch_size, shuffle=False)

    train(model=model, optimizer=optimizer, criteria=criteria, train_iter=train_iter, dev_iter=dev_iter,
          output_dir=output_path, device=device, epoch_num=epoch_num)


if __name__ == '__main__':
    main()
    print('')






