import pandas as pd
import csv
from collections import Counter

data_file_path='/home/panhui/PycharmProjects/SAN-SAW/datasets/assets/val_out.csv'
pd_data = pd.read_csv(data_file_path)
image_path_list = list(pd_data['img_pth'])
source_list = list(pd_data['source'])
image_label_list = []
print(Counter(source_list))
