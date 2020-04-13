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
from torch.utils.data.sampler import BatchSampler
from collections import defaultdict

class Dataset(data.Dataset):

    def __init__(self, root, pd_root, data_list_file, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        self.imgs = np.random.permutation(imgs)
        self.pdimage =[os.path.join(pd_root, img[:-1]) for img in self.imgs]

        self.train_data = [os.path.join(root, img.split()[0]) for img in imgs]
        self.train_labels = [np.int32(img.split()[1]) for img in imgs]
        self.train_pddata = [os.path.join(pd_root, img.split()[0]) for img in imgs]

        self.train = self.phase

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.CenterCrop([160, 160]),
                T.ToTensor(),
                normalize
            ])
            self.pd_transforms = T.Compose([
                T.CenterCrop([160, 160]),
                T.ToTensor()
            ])

        else:
            self.transforms = T.Compose([
                T.CenterCrop([160, 160]),
                T.ToTensor(),
                normalize
            ])

            self.pd_transforms = T.Compose([
                T.CenterCrop([160, 160]),
                T.ToTensor()
            ])



    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        data = data.convert('L')
        data = self.transforms(data)
        label = np.int32(splits[1])

        sample = self.pdimage[index]
        splits = sample.split()
        img_path = splits[0]
        pddata = Image.open(img_path)
        pddata = pddata.convert('L')
        pddata = self.pd_transforms(pddata)


        return data.float(), pddata.float(), label

    def __len__(self):
        return len(self.imgs)


class SiameseData(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset

        self.train = self.dataset.train
        self.transform = self.dataset.transforms
        self.pd_transforms = self.dataset.pd_transforms

        if self.train:
            self.train_labels = self.dataset.train_labels
            self.train_data = self.dataset.train_data
            self.train_pddata = self.dataset.train_pddata

            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.dataset.test_labels
            self.test_data = self.dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            # target = 1  ################
            img1, pdimg1,  label1 = self.train_data[index], self.train_pddata[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2, pdimg2,  label2 = self.train_data[siamese_index], self.train_pddata[siamese_index], self.train_labels[siamese_index].item()
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.open(img1)
        img1 = img1.convert('L')
        img2 = Image.open(img2)
        img2 = img2.convert('L')
        pdimg1 = Image.open(pdimg1)
        pdimg1 = pdimg1.convert('L')

        pdimg2 = Image.open(pdimg2)
        pdimg2 = pdimg2.convert('L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

            pdimg1 = self.pd_transforms(pdimg1)
            pdimg2 = self.pd_transforms(pdimg2)
            pdimg1 = pdimg1[:,1::4,1::4]
            pdimg2 = pdimg2[:, 1::4, 1::4]
            pdimg1[pdimg1 > 165 / 256] = pdimg1[pdimg1 > 165 / 256] - 1
            pdimg1[pdimg1 > 90 / 256] = 180/256
            pdimg1 = torch.cat((torch.cos(pdimg1*256/180*math.pi),  torch.sin(pdimg1*256/180*math.pi)), 0)
            pdimg2[pdimg2 > 165 / 256] = pdimg2[pdimg2 > 165 / 256] - 1
            pdimg2[pdimg2 > 90 / 256] = 180 / 256
            pdimg2 = torch.cat((torch.cos(pdimg2*256/180*math.pi), torch.sin(pdimg2*256/ 180 * math.pi)),0)

        return img1, img2, (pdimg1, pdimg2), (label1, label2)

    def __len__(self):
        return len(self.mnist_dataset)


class BatchGenerator(pdimg):
    def __init__(self, labels, num_instances, batch_size):
        self.labels = labels
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.ids = set(self.labels)
        self.num_id = batch_size//num_instances

        self.index_dic = defaultdict(list)

        for index, cat_id in enumerate(self.labels):
            self.index_dic[cat_id].append(index)

    def __len__(self):
        return self.num_id*self.num_instances

    def batch(self):
        ret = []
        indices = np.random.choice( list(self.ids), size=self.num_id, replace=False)
        # print(indices)
        for cat_id in indices:
            t = self.index_dic[cat_id]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return ret

    def get_id(self):
        ret = self.batch()
        # print(ret)
        result = [self.labels[k] for k in ret]
        return result




class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples, batch_size):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        classes = np.random.choice(self.labels_set, self.n_classes, replace=False)

        num = int(self.batch_size/self.n_samples)
        while self.count + self.batch_size < self.n_dataset:
            indices = []
            l = 0
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                if indices.length>self.batch_size:
                    yield indices
            self.count += self.batch_size


    def __len__(self):
        return self.n_dataset // self.batch_size

