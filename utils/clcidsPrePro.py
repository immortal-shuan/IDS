import os
import pandas as pd


data_path = 'F:\data\CIC-IDS-2017\MachineLearningCSV'
file_name = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
             'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
             'Friday-WorkingHours-Morning.pcap_ISCX.csv',
             'Monday-WorkingHours.pcap_ISCX.csv',
             'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
             'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
             'Tuesday-WorkingHours.pcap_ISCX.csv',
             'Wednesday-workingHours.pcap_ISCX.csv']

data_path1 = os.path.join(data_path, file_name[0])
data1 = pd.read_csv(data_path1)

b = data1.loc[:, [' Label']].values.reshape(-1).tolist()








