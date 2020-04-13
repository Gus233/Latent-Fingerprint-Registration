import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys

import numpy as np
from PIL import Image
import math

from torch.utils.data import Dataset
from collections import defaultdict

class Dataset(data.Dataset):

    def __init__(self, root, pd_root,  data_list_file, phase='train', input_shape=(1, 160, 160)):
        self.phase = phase
        self.input_shape = input_shape
        self.root = root
        self.pd_root = pd_root
        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        self.imgs = np.random.permutation(imgs)
        self.train = self.phase

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])
            self.pd_transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
            ])

        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])
            self.pd_transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
            ])




    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = os.path.join(self.root, splits[0])
        data = Image.open(img_path)
        data = data.convert('L')
        data = self.transforms(data)
        image1 = data

        img_path = os.path.join(self.root, splits[1])
        data = Image.open(img_path)
        data = data.convert('L')
        data = self.transforms(data)
        image2 = data


        img_path = os.path.join(self.pd_root, splits[0])
        pdimg = Image.open(img_path)
        pdimg = pdimg.convert('L')
        pdimg1 = self.pd_transforms(pdimg)

        img_path = os.path.join(self.pd_root, splits[1])
        pdimg = Image.open(img_path)
        pdimg = pdimg.convert('L')
        pdimg2 = self.pd_transforms(pdimg)

        pdimg1 = pdimg1[:, 3::8, 3::8]
        pdimg2 = pdimg2[:, 3::8, 3::8]
        pdimg1[pdimg1 > 165 / 256] = pdimg1[pdimg1 > 165 / 256] - 1
        pdimg1[pdimg1 > 90 / 256] = 180 / 256
        pdimg1 = torch.cat((torch.cos(pdimg1 * 256 / 180 * math.pi), torch.sin(pdimg1 * 256 / 180 * math.pi)), 0)
        pdimg2[pdimg2 > 165 / 256] = pdimg2[pdimg2 > 165 / 256] - 1
        pdimg2[pdimg2 > 90 / 256] = 180 / 256
        pdimg2 = torch.cat((torch.cos(pdimg2 * 256 / 180 * math.pi), torch.sin(pdimg2 * 256 / 180 * math.pi)), 0)


        dx = np.int32(splits[2])
        dy = np.int32(splits[3])
        da = np.int32(splits[4])

        label = torch.FloatTensor([dx/300.0, dy/300.0, da/40.0])

        return image1.float(), image2.float(), (pdimg1.float(), pdimg2.float()), label

    def __len__(self):
        return len(self.imgs)

