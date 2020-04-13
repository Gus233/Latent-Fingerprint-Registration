

from __future__ import print_function
import os
import cv2
import torch
import time
import numpy as np
from scipy import io
from torch.nn import DataParallel

from models import *

from scipy import io, misc
from matplotlib import  pyplot as plt
from torch.nn.modules.module import Module
from torch.utils import data

from dataset import TestDataset

import torch.nn.functional as F

class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch=3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)




def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    return pairs


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    oimage = image

    oimage = oimage[np.newaxis, :, :]
    oimage = oimage.astype(np.float32, copy=False)


    image = image[40:200, 40:200]
    image = image[np.newaxis, np.newaxis, :, :]

    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image, oimage

def get_featurs(lfw_root, pairs,  register_model, similarity_model, save_path):

    batch_size = 20
    transs = None
    distances = None
    images1 = None
    images2 = None
    #oimages1 = None
    oimages2 = None
    cnt = 0

    for pair in pairs:
        cnt = cnt + 1
        data_path1 = os.path.join(lfw_root, pair.split()[0])
        data_path2 = os.path.join(lfw_root, pair.split()[1])

        image1, oimage1 = load_image(data_path1)
        image2, oimage2 = load_image(data_path2)

        if images1 is None:
            images1 = image1
            images2 = image2
          #  oimages1 = oimage1
            oimages2 = oimage2
        else:
            images1 = np.concatenate((images1, image1), axis=0)
            images2 = np.concatenate((images2, image2), axis=0)
          #  oimages1 = np.concatenate((oimages1, oimage1), axis=0)
            oimages2 = np.concatenate((oimages2, oimage2), axis=0)

        if images1.shape[0] % batch_size == 0 or cnt == len(pairs):
        #    start = time.time()

            images1 = torch.from_numpy(images1).cuda()
            images2 = torch.from_numpy(images2).cuda()

            data = tuple((images1, images2))

       #     data = (torch.from_numpy(images1), torch.from_numpy(images2))
       #     data = tuple(d.cuda() for d in data)
        #    print('Speed:{}'.format((time.time() - start) / batch_size))

            output = register_model(*data)


            trans = output[0].data#.cpu().numpy()
            if transs is None:
                transs = trans
            else:
                transs = torch.cat((transs, trans), 0)

            para = output[0].data
            alpha = para[:,2]  #*np.pi/180
            theta =  torch.cat((para, para), 1)
            theta[:,2] = -para[:,0]*40.0/120.0
            theta[:,5] = -para[:,1]*40.0/120.0
            theta[:,0] = np.cos(alpha)
            theta[:,1] = -np.sin(alpha)
            theta[:,3] = np.sin(alpha)
            theta[:,4] = np.cos(alpha)
            theta = theta.reshape((para.shape[0], 2, 3))

            sampling_grid = AffineGridGen(240,240,1)(theta)
            tmp = torch.from_numpy(255.0-oimages2[:, np.newaxis,:,:])
            tmp = tmp.cuda()

            warped_image_batch = F.grid_sample(tmp,  sampling_grid)
            rec_images2 = (255.0-warped_image_batch-127.5)/ 127.5
            rec_images2 = rec_images2[:,:,40:200, 40:200]

            data = tuple((images1, rec_images2))

            output = similarity_model(*data)
            distance = F.cosine_similarity(output[0], output[1])
            distance = distance.data

            if distances is None:
                distances = distance
            else:
                distances = torch.cat((distances, distance), 0)

            images1 = None
            images2 = None
     #       oimages1 = None
            oimages2 = None


        if cnt % 1000 == 0:
            message = '[{}/{} ({:.0f}%)]'.format(
            cnt, len(pairs), 100. * cnt  / len(pairs))

            print(message)

    transs = transs.cpu().numpy()
    distances = distances.cpu().numpy()
    return transs, distances


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def lfw_test(test_root, pairs,  register_model, similarity_model, save_path, trans_save_path,  sim_save_path):
    transs, sims = get_featurs(test_root, pairs,  register_model, similarity_model,  save_path)
    io.savemat(trans_save_path, {'trans': transs})
    io.savemat(sim_save_path, {'sims': sims})


if __name__ == '__main__':
    register_model_path = '/home/data2/gus/LatentMatch/checkpoints/ave_40/select/resnet18_20.pth'   # threshold indeed 0.85'   #  select2
    #ave_40_wo_roi/combine2/resnet18_20.pth'   # select1'

    similarity_model_path = '/home/data2/gus/MinutiaeDescriptor/joint/checkpoints/create/32_new_160_add_texture_wo_roi/image_select80/resnet18_20.pth'  # select2'
    # 32_new_160_add_texture_wo_roi/resnet18_20.pth'  # select1


    test_root = '/home/data2/gus/LatentMatch/nist27/gt_seg/all_pair/test_patch_pair_NIST27/'
    list_path = '/home/data2/gus/LatentMatch/nist27/gt_seg/all_pair/test_patch_pair_NIST27_list/'  # tmp_pair.txt'
    image_save_path = '/home/data2/gus/LatentMatch/nist27/afis_seg/'
    trans_save_path = '/home/data2/gus/LatentMatch/nist27/gt_seg/all_pair/test_result/'
    sim_save_path =  trans_save_path

    if not os.path.exists(trans_save_path):
        os.makedirs(trans_save_path)


    test_batch_size = 10


    os.environ["CUDA_VISIBLE_DEVICES"] =  '3'
    embedding_net = Register_Unet_down()
    regression_net = Register_Unet_up()
    model = Register_SiameseNet(embedding_net, regression_net)
    model = DataParallel(model)
    load_model(model, register_model_path)
    model.load_state_dict(torch.load(register_model_path))
    model.to(torch.device("cuda"))
    register_model = model
    register_model.eval()

    embedding_net = Similarity_Unet_down()
    regression_net = Similarity_Unet_up()
    model = Similarity_SiameseNet(embedding_net, regression_net)
    model = DataParallel(model)
    load_model(model, similarity_model_path)
    model.load_state_dict(torch.load(similarity_model_path))
    model.to(torch.device("cuda"))
    similarity_model = model
    similarity_model.eval()

    for i in range(1):
        for j in range(1):
            ii = i+1
            jj = j+1

            test_list = os.path.join(list_path+str(ii), str(ii)+'_'+str(jj)+'.txt')
            pairs = get_lfw_list(test_list)
            single_image_save_path = os.path.join(image_save_path, str(ii)+'_'+str(jj)+'/')

            if not os.path.exists(trans_save_path + str(ii)):
                os.makedirs(trans_save_path + str(ii))
            single_trans_save_path = os.path.join(trans_save_path+str(ii), str(ii) + '_' + str(jj)+'_trans.mat')
            single_sims_save_path = os.path.join(sim_save_path+str(ii), str(ii)+'_'+str(jj)+'_sims.mat')

            start = time.time()
            lfw_test(test_root, pairs,  register_model, similarity_model, single_image_save_path, single_trans_save_path,  single_sims_save_path)
            time_str = time.asctime(time.localtime(time.time()))
            print('{} Current: {}_{} Speed:{}'.format(time_str, ii, jj, (time.time() - start)/len(pairs)))




