
import time
import mmcv
import torch
import numpy as np
import os

import torch.nn.functional as F
from utils.train_helper import get_model
from utils.util import Split_6, Merge_6

class SS_GluePathInspection(object):
    def __init__(self, args, ):
        self.args = args
        self.black_img = np.zeros([3648, 5472], dtype=np.uint8)

        self.mean = [123.675, 116.28 , 103.53 ]
        self.std = [58.395, 57.12 , 57.375]

    def load_model(self, filename,param):
        self.model, params = get_model(self.args)
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        checkpoint = torch.load(filename)

        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.module.load_state_dict(checkpoint)

        self.model.eval()
        self.model.module.eval()
        self.param =  param
        return 0


    def predict(self, cv_img):
        seg_img = cv_img[..., ::-1]

        dict_part2img = Split_6(seg_img, self.param)
        dict_part2mask1={}
        dict_part2mask2={}
        dict_part2mask3={}
        for part,img in dict_part2img.items():
            _mask1,_mask2,_mask3 = self.forward(img)
            if part == 'l' or part == 'r':
                _mask1,_mask2 ,_mask3= np.rot90(_mask1, -1), np.rot90(_mask2, -1), np.rot90(_mask3, -1)
            dict_part2mask1[part] = (_mask1*255).astype(np.uint8)
            dict_part2mask2[part] = (_mask2*255).astype(np.uint8)
            dict_part2mask3[part] = (_mask3*255).astype(np.uint8)

        mask1 = Merge_6(self.black_img, dict_part2mask1, self.param)
        mask2 = Merge_6(self.black_img, dict_part2mask2, self.param)
        mask3 = Merge_6(self.black_img, dict_part2mask3, self.param)

        return mask1, mask2, mask3

    def forward(self, img):
        img = ((np.array(img) - self.mean) / self.std).astype(dtype=np.float32, copy=True)
        img = img[np.newaxis, :]
        img = torch.from_numpy(img.copy())
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img.to("cuda")
        with torch.no_grad():
            preds = self.model(img)
            if isinstance(preds, tuple):

                pred = preds[0]
                pred_lay2 = preds[2]
                pred_lay1 = preds[3]

                pred_P = pred.max(1)[1]
                pred_lay2 = pred_lay2.max(1)[1]
                pred_lay1 = pred_lay1.max(1)[1]

                pred_P = pred_P.detach().cpu().numpy()[0]
                pred_lay2 = pred_lay2.detach().cpu().numpy()[0]
                pred_lay1 = pred_lay1.detach().cpu().numpy()[0]

        return pred_P,pred_lay1,pred_lay2


dict_param={}
dict_param['roi_l'] = [180, 716, 0 + 360, 0 + 2640]
dict_param['roi_r'] = [4850, 716, 0 + 360, 0 + 2640]
dict_param['roi_tl'] = [220, 786, 0 + 2640, 0 + 360]
dict_param['roi_bl'] = [220, 2890, 0 + 2640, 0 + 360]
dict_param['roi_tr'] = [2532, 786, 0 + 2640, 0 + 360]
dict_param['roi_br'] = [2532, 2890, 0 + 2640, 0 + 360]



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def add_train_args(arg_parser):
    # train related arguments
    arg_parser.add_argument('--seed', default=12345, type=int,
                            help='random seed')
    arg_parser.add_argument('--gpu', type=str, default="0",
                            help=" the num of gpu")
    arg_parser.add_argument('--batch_size_per_gpu', default=4, type=int,
                            help='input batch size')
    arg_parser.add_argument('--alpha', default=0.3, type=int,
                            help='input mix alpha')

    # dataset related arguments
    arg_parser.add_argument('--num_classes', default=2, type=int,
                            help='num class of mask')
    arg_parser.add_argument('--data_loader_workers', default=1, type=int,
                            help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')

    # optimization related arguments

    arg_parser.add_argument('--optim', default="SGD", type=str,
                            help='optimizer')
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)

    arg_parser.add_argument('--lr', type=float, default=5e-4,
                            help="init learning rate ")
    arg_parser.add_argument('--iter_max', type=int, default=200000,
                            help="the maxinum of iteration")
    arg_parser.add_argument('--iter_stop', type=int, default=200000,
                            help="the early stop step")
    arg_parser.add_argument('--each_epoch_iters', default=10000,
                            help="the path of ckpt file")
    arg_parser.add_argument('--poly_power', type=float, default=0.9,
                            help="poly_power")
    arg_parser.add_argument('--selected_classes', default=[ 1],
                            help="poly_power")
    arg_parser.add_argument('--weight_loss', default=False,
                            help="if use weight loss")
    arg_parser.add_argument('--use_trained', default=False,
                            help="if use trained model")
    arg_parser.add_argument('--backbone', default='Deeplab50_CLASS_INW',
                            help="backbone of encoder")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply imagenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether apply pretrained checkpoint")
    arg_parser.add_argument('--continue_training', type=str2bool, default=False,
                            help="whether to continue training ")
    return arg_parser


import cv2,argparse
from tqdm import tqdm
arg_parser = argparse.ArgumentParser()
arg_parser = add_train_args(arg_parser)

args = arg_parser.parse_args()

seg = SS_GluePathInspection(args)
seg.load_model('/home/panhui/PycharmProjects/SAN-SAW/model/gluebest.pth',dict_param)

data_dir= 'Dubai_0708'
img_dir = '/media/data1/panhui9/dianjiao_data/tmp/{}/'.format(data_dir)
vis_dir = 'test'
os.makedirs(vis_dir, exist_ok=True)
for img_name in tqdm(sorted(os.listdir(img_dir), reverse=True)):
    img_path = os.path.join(img_dir, img_name)
    # if os.path.exists(os.path.join(vis_dir, img_name.replace('.jpg', '.png'))):
    #     continue
    img = cv2.imread(img_path)
    r = seg.predict(img)
    cv2.imwrite(os.path.join(vis_dir, img_name.replace('.jpg', '.png')), np.uint8(r[0]))
    cv2.imwrite(os.path.join(vis_dir, img_name.replace('.jpg', '_1.png')), np.uint8(r[1]))
    cv2.imwrite(os.path.join(vis_dir, img_name.replace('.jpg', '_2.png')), np.uint8(r[2]))