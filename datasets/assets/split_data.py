import pandas as pd
import csv
data_file_path='./train_tonga_1022.csv'
pd_data = pd.read_csv(data_file_path)
image_path_list = list(pd_data['img_pth'])
source_list = list(pd_data['source'])
image_label_list = []
sub_class2passfilelist={}
for i, img_path in enumerate(image_path_list):
    data_source = source_list[i]
    image_name = img_path.split('/')[-1]
    item = (img_path, data_source)
    image_label_list.append(item)

    raw_img_name = image_name.rsplit('_', 1)[0]
    if data_source not in sub_class2passfilelist:
        sub_class2passfilelist[data_source]=[]
    sub_class2passfilelist[data_source].append(img_path)

import random
import numpy as np
train_list = []
val_list = []
for ds in sub_class2passfilelist.keys():
    num_img = len(sub_class2passfilelist[ds])
    ds_pd_data = pd_data[pd_data['source']==ds]
    assert num_img == len(ds_pd_data),'{}，{}，{}'.format(num_img,len(ds_pd_data),ds)
    val_imgs = random.sample(sub_class2passfilelist[ds], int(np.ceil(num_img/10)))
    val_data = ds_pd_data[ds_pd_data['img_pth'].isin(val_imgs)]
    train_data = ds_pd_data[~ds_pd_data['img_pth'].isin(val_imgs)]
    train_list.append(train_data)
    val_list.append(val_data)

new_data0 = pd.concat(train_list)
new_data1 = pd.concat(val_list)
new_data0.to_csv('train_1027.csv', index=False)
new_data1.to_csv('val_1027.csv', index=False)