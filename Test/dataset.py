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


class  TestDataset(data.Dataset):



    def __init__(self, root, data_list_file, phase='test', input_shape=(1, 160,160)):

        self.phase = phase
        self.input_shape = input_shape
        self.root = root


        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        self.imgs = imgs

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'test':
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])
            self.another_transforms = T.Compose([
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
        oimage2 = self.another_transforms(data)
        data = self.transforms(data)
        image2 = data

        return image1.float(), image2.float(), oimage2.float()

    def __len__(self):
        return len(self.imgs)


