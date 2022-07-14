"""
Cityscapes Dataset Loader
"""
import logging
import json
import os
import numpy as np
from PIL import Image, ImageCms

from torch.utils import data
import torch

import torchvision.transforms as ttransforms
import pandas as pd
from datasets.transforms import  joint_transforms as joint_transforms
from datasets.transforms import transforms as extended_transforms
num_classes = 2
ignore_label = 255

from torch.utils.data import DataLoader, ConcatDataset


def make_dj_dataset_from_csv(data_file_path, data_source_list):
    if not os.path.isfile(data_file_path):
        raise (RuntimeError("Image list file do not exist: " + data_file_path + "\n"))
    pd_data = pd.read_csv(data_file_path)
    image_path_list = list(pd_data['img_pth'])
    source_list = list(pd_data['source'])
#     print("Processing data...".format(data_file_path))
    image_label_list = []

    for i, img_path in enumerate(image_path_list):
        data_source = source_list[i]
        if 1:
            label_path = img_path.replace('.jpg', '.png')
            item = (img_path, label_path)
            image_label_list.append(item)

#     print("Checking image&label {} list done! ".format(data_file_path))
    return image_label_list

class Glue(data.Dataset):

    def __init__(self, data_csv_path, data_source_list, joint_transform=None,
                 transform=None, target_transform=None, target_aux_transform=None,
                 image_in=False, extract_feature=False):
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.target_aux_transform = target_aux_transform
        self.image_in = image_in
        self.extract_feature = extract_feature
        self.imgs = make_dj_dataset_from_csv(data_csv_path, data_source_list)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        #         print('0',img.size,mask.size,len(mask.split()))
        if len(mask.split()) >=3:
            mask = mask.split()[0]
        #             print('1',img.size,mask.size,len(mask.split()))

        if img.size[0] < img.size[1]:  # rotate 90
            img = img.transpose(Image.ROTATE_90)
        if mask.size[0] < mask.size[1]:
            mask = mask.transpose(Image.ROTATE_90)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # print('1',img.size,mask.size)
        mask = np.array(mask)

        mask = np.where(mask == 255, 1, 0)
        mask = Image.fromarray(mask.astype(np.uint8))

        # Image Transformations
        try:
            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)
        except:
            print(img_path, mask_path,img.size,mask.size)
        if self.transform is not None:
            img = self.transform(img)

        img = ttransforms.Normalize(*self.rgb_mean_std)(img)
        if self.target_aux_transform is not None:
            mask_aux = self.target_aux_transform(mask)
        else:
            mask_aux = torch.tensor([0])
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        #         print(img.size(),mask.size(),img_name)
        return img, mask, img_path

    def __len__(self):
        return len(self.imgs)

def get_train_joint_transform(args):
    """
    Get train joint transform
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_joint_transform_list, train_joint_transform
    """

    # Geometric image transformations
    train_joint_transform_list = []
    if args.glue:
        train_joint_transform_list += [
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomVerticallyFlip(),
        ]
    else:
        train_joint_transform_list += [
            joint_transforms.RandomSizeAndCrop(args.crop_size,
                                               crop_nopad=args.crop_nopad,
                                               pre_size=args.pre_size,
                                               scale_min=args.scale_min,
                                               scale_max=args.scale_max,
                                               ignore_index=-1),
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip()]

    if args.rrotate > 0:
        train_joint_transform_list += [joint_transforms.RandomRotate(
            degree=args.rrotate,
            ignore_index=-1)]

    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # return the raw list for class uniform sampling
    return train_joint_transform_list, train_joint_transform

def get_input_transforms(args):
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """
    # Image appearance transformations
    train_input_transform = []
    val_input_transform = []
    if args.glue:
        train_input_transform += [
            # extended_transforms.RandomContrast(),
            # extended_transforms.AdditiveNoise(),
            ttransforms.RandomApply([
                ttransforms.ColorJitter(0.2, 0.2, 0.1, 0.1)],
                p=0.5)
        ]
    else:
        if args.color_aug > 0.0:
            train_input_transform += [ttransforms.RandomApply([
                ttransforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)]

        if args.bblur:
            train_input_transform += [extended_transforms.RandomBilateralBlur()]
        elif args.gblur:
            train_input_transform += [extended_transforms.RandomGaussianBlur()]

    train_input_transform += [
        ttransforms.ToTensor()
    ]
    val_input_transform += [
        ttransforms.ToTensor()
    ]
    train_input_transform = ttransforms.Compose(train_input_transform)
    val_input_transform = ttransforms.Compose(val_input_transform)

    return train_input_transform, val_input_transform

def get_target_transforms(args):
    """
    Get target transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: target_transform, target_train_transform, target_aux_train_transform
    """

    target_transform = extended_transforms.MaskToTensor()
    if 0:
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(
            -1, 2)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    target_aux_train_transform = extended_transforms.MaskToTensor()

    return target_transform, target_train_transform, target_aux_train_transform


class Glue_DataLoader():
    def __init__(self, args, training=True):
        args.glue = True
        self.args = args

        train_joint_transform_list, train_joint_transform = get_train_joint_transform(args)
        train_input_transform, val_input_transform = get_input_transforms(args)
        target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args)

        train_sets = Glue('./datasets/assets/train.csv', args.train_data_list,
                                         joint_transform=train_joint_transform,
                                         transform=train_input_transform,
                                         target_transform=target_train_transform,
                                         target_aux_transform=target_aux_train_transform,
                                         image_in=False)
        
        val_sets = Glue('./datasets/assets/val.csv', args.val_data_list,
                       transform=val_input_transform,
                       target_transform=target_transform,
                       image_in=False)
        val_sets1 = Glue('./datasets/assets/val_out.csv', args.val_data_list,
                       transform=val_input_transform,
                       target_transform=target_transform,
                       image_in=False)
        print('train_sets',len(train_sets))
                
        if len(train_sets) == 0:
            raise Exception('Dataset {} is not supported'.format(args.dataset))

        # Define new train data set that has all the train sets
        # Define new val data set that has all the val sets
        
        
#         if len(train_sets) != 1:
#             train_sets = ConcatDataset(train_sets)
#         if len(val_sets) != 1:
#             val_sets = ConcatDataset(val_sets)
        bs =4 
        self.data_loader = data.DataLoader(train_sets,
                                               batch_size= bs,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
      

        self.val_loader = data.DataLoader(val_sets,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=self.args.data_loader_workers,
                                          pin_memory=self.args.pin_memory,
                                          drop_last=True)

        self.val_loader1 = data.DataLoader(val_sets1,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=self.args.data_loader_workers,
                                          pin_memory=self.args.pin_memory,
                                          drop_last=True)
        print('len(val_sets)',len(val_sets))
        self.valid_iterations = (len(val_sets) + 1) // 1
        self.valid_iterations1 = (len(val_sets1) + 1) // 1

        self.num_iterations = (len(train_sets) + 1) // bs


class GlueAug(data.Dataset):

    def __init__(self, data_csv_path, data_source,
                 transform=None, color_transform=None, geometric_transform=None,
                 target_transform=None, image_in=False, extract_feature=False):

        self.transform = transform
        self.target_transform = target_transform
        self.color_transform = color_transform
        self.geometric_transform = geometric_transform
        self.image_in = image_in
        self.extract_feature = extract_feature
        self.imgs = make_dj_dataset_from_csv(data_csv_path, [data_source])
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        #         print('0',img.size,mask.size,len(mask.split()))
        if len(mask.split()) == 4:
            mask = mask.split()[0]
            #             print('1',img.size,mask.size,len(mask.split()))
        if img.size[0] < img.size[1]:  # rotate 90
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # print('1',img.size,mask.size)
        mask = np.array(mask)

        mask = np.where(mask == 255, 1, 0)
        mask = Image.fromarray(mask.astype(np.uint8))

        # # Image Transformations
        # if self.extract_feature is not True:
        #     if self.joint_transform is not None:
        #         img, mask = self.joint_transform(img, mask)

        if self.transform is not None:
            img_or = self.transform(img)

        if self.color_transform is not None:
            img_color = self.color_transform(img)

        if self.geometric_transform is not None:
            img_geometric = self.geometric_transform(img)

        rgb_mean_std_or = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        rgb_mean_std_color = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        rgb_mean_std_geometric = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.image_in:
            eps = 1e-5
            rgb_mean_std_or = ([torch.mean(img_or[0]), torch.mean(img_or[1]), torch.mean(img_or[2])],
                               [torch.std(img_or[0]) + eps, torch.std(img_or[1]) + eps, torch.std(img_or[2]) + eps])
            rgb_mean_std_color = ([torch.mean(img_color[0]), torch.mean(img_color[1]), torch.mean(img_color[2])],
                                  [torch.std(img_color[0]) + eps, torch.std(img_color[1]) + eps,
                                   torch.std(img_color[2]) + eps])
            rgb_mean_std_geometric = (
                [torch.mean(img_geometric[0]), torch.mean(img_geometric[1]), torch.mean(img_geometric[2])],
                [torch.std(img_geometric[0]) + eps, torch.std(img_geometric[1]) + eps,
                 torch.std(img_geometric[2]) + eps])
        img_or = ttransforms.Normalize(*rgb_mean_std_or)(img_or)
        img_color = ttransforms.Normalize(*rgb_mean_std_color)(img_color)
        img_geometric = ttransforms.Normalize(*rgb_mean_std_geometric)(img_geometric)

        return img_or, img_color, img_geometric, img_path

    def __len__(self):
        return len(self.imgs)
