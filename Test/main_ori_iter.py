
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
import scipy.io as scio
import scipy.ndimage
import torch.nn.functional as F
import h5py
import pdb

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

    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def load_roi(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None

    image = image.astype(np.float32, copy=False)

    image /= 255.0
    return image

def cut_image(img, posx, posy, width, padsize):
    img = torch.unsqueeze(img,0)
    img = torch.unsqueeze(img, 0)

    posx = int(posx + padsize)
    posy = int(posy+  padsize)
    image = img[:,:,posy-width:posy+width, posx-width:posx+width]

    return image

def get_featurs(I1, I2,  pairs,  patchsize, register_model, similarity_model):

    batch_size = 20
    transs = None
    distances = None
    oimages1 = None
 #   images2 = None
    oimages2 = None
    ori = None
    cnt = 0


    maxsize = patchsize + 50

    I1 = np.pad(I1,((maxsize, maxsize),(maxsize, maxsize)), 'constant', constant_values = 1.0)
    I2 = np.pad(I2, ((maxsize), (maxsize)), 'constant', constant_values=1.0)
    I1 = torch.from_numpy(I1).cuda()
    I2 = torch.from_numpy(I2).cuda()



    for pair in pairs:

        cnt = cnt + 1

        splits = pair.split()
        oimage1 = cut_image(I1, int(splits[0]), int(splits[1]), maxsize, maxsize)

        oimage2 = cut_image(I2, int(splits[2]), int(splits[3]), maxsize, maxsize)


        if oimages1 is None:
            oimages1 = oimage1
        #    images2 = image2
            oimages2 = oimage2
            ori = torch.tensor([int(splits[4])])
        else:
            oimages1 = torch.cat((oimages1, oimage1), 0)
        #    images2 = torch.cat((images2, image2), 0)

            oimages2 = torch.cat((oimages2, oimage2),0)

            ori = torch.cat((ori, torch.tensor([int(splits[4])])), 0)


        images2 = oimages2[:,:,50:maxsize*2-50,50:maxsize*2-50]
        images1 = oimages1[:, :, 50:maxsize*2-50, 50:maxsize*2-50]
        del oimage1, oimage2




        if images1.shape[0] % batch_size == 0 or cnt == len(pairs):
        #    start = time.time()

            # ------------ step 1  predict the trans  ----- 0.0007
            output = register_model(images1, images2)
            trans = output[0].data
            if transs is None:
                transs = trans
            else:
                transs = torch.cat((transs, trans), 0)

            #-------------- step 2  rec image patch  ---0.0005
            para = output[0].data
            ori = ori.cuda().float()
            alpha = -ori / 180.0 * 3.1415926  #
            theta = torch.cat((para, para), 1)
            theta[:, 2] = 0.0
            theta[:, 5] = 0.0
            theta[:, 0] = torch.cos(alpha)
            theta[:, 1] = -torch.sin(alpha)
            theta[:, 3] = torch.sin(alpha)
            theta[:, 4] = torch.cos(alpha)
            theta = theta.reshape((para.shape[0], 2, 3))
            sampling_grid = AffineGridGen(maxsize * 2, maxsize * 2, 1)(theta)
            warped_image_batch = F.grid_sample(1.0 - oimages1, sampling_grid)
            rec_images1 = (1.0 - warped_image_batch)
            rec_images1 = rec_images1[:, :, 50:maxsize * 2 - 50, 50:maxsize * 2 - 50]


            para = output[0].data
            alpha = (para[: ,2]* 20-ori)/180*3.1415926
            theta =  torch.cat((para, para), 1)
            theta[: ,2] = -para[: ,0 ] *25.0/maxsize
            theta[: ,5] = -para[: ,1 ] *25.0/maxsize
            theta[: ,0] = torch.cos(alpha)
            theta[: ,1] = -torch.sin(alpha)
            theta[: ,3] = torch.sin(alpha)
            theta[: ,4] = torch.cos(alpha)
            theta = theta.reshape((para.shape[0], 2, 3))

            sampling_grid = AffineGridGen(maxsize*2, maxsize*2 ,1)(theta)
            warped_image_batch = F.grid_sample( 1.0 - oimages2,  sampling_grid)
            rec_images2 = (1.0-warped_image_batch)
            rec_images2 = rec_images2[: ,: ,50:maxsize*2-50,50:maxsize*2-50]







            del theta, output,alpha#,para
            del sampling_grid, warped_image_batch
          #  del images2, oimages2



          # --------  step 3   similarity -------------------------------  0.0017


            output1 = similarity_model.module.get_embedding(rec_images1)
            output2 = similarity_model.module.get_embedding(rec_images2)

            distance = F.cosine_similarity(output1, output2)
            distance = distance.data

            del  output1, output2 #, rec_images2


            if distances is None:
                distances = distance
            else:
                distances = torch.cat((distances, distance), 0)


            oimages1 = None
          #  images2 = None
            oimages2 = None
            ori = None

        if cnt % 1000 == 0:
            message = '[{}/{} ({:.0f}%)]'.format(
                cnt, len(pairs), 100. * cnt  / len(pairs))

            print(message)
    flag = 0
    if len(pairs) > 0:
        transs = transs.cpu().numpy()
        distances = distances.cpu().numpy()
        flag = 1
    return transs, distances, flag


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def lfw_test(image1, image2, pairs,  patchsize, register_model, similarity_model, trans_save_path,  sim_save_path):

    transs, sims, flag = get_featurs(image1, image2, pairs, patchsize, register_model, similarity_model)

    if flag==1:

        io.savemat(trans_save_path, {'trans': transs})
        io.savemat(sim_save_path, {'sims': sims})


if __name__ == '__main__':
    register_model_path = '/align25.pth'
    similarity_model_path = '/descriptor.pth'  

    image_path = '/media/disk3/gs/LatentMatchData/TestTime/align3_des6/test_NIST27_image_register/'
    minu_path = '/media/disk3/gs/LatentMatchData//NIST27_iter_minu_24x24_pair/'

    trans_save_path = '/media/disk3/gs/LatentMatchData/result_ori_iter_24/'
    sim_save_path =  trans_save_path
    if not os.path.exists(trans_save_path):
        os.makedirs(trans_save_path)

    os.environ["CUDA_VISIBLE_DEVICES"] =  '1'
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

    patchsize = 100



    for i in range(258):
        for j in range(i,i+1):
            ii = i+1
            jj = i+1
            print(ii)

            data_path1 = os.path.join(image_path , str(ii) + '_' + str(jj) + '_1.jpg')
            image1 = load_image(data_path1)
            data_path2 = os.path.join(image_path , str(ii) + '_' + str(jj) + '_2.jpg')
            image2 = load_image(data_path2)

            test_minu = os.path.join(minu_path,  str(ii) + '_' + str(jj) + '.txt')
            pairs = get_lfw_list(test_minu)


#            single_minus_save_path = os.path.join(trans_save_path, str(ii) + '_' + str(jj) + '_minus.mat')
#            io.savemat(single_minus_save_path , {'minu1': minu1, 'minu2':minu2})
            if len(pairs) > 0:
                single_trans_save_path = os.path.join(trans_save_path, str(ii) + '_' + str(jj) + '_trans.mat')
                single_sims_save_path = os.path.join(sim_save_path, str(ii) + '_' + str(jj) + '_sims.mat')

                start = time.time()
                lfw_test(image1, image2,  pairs, patchsize, register_model, similarity_model, single_trans_save_path, single_sims_save_path)
                time_str = time.asctime(time.localtime(time.time()))

                print('{} Current: {}_{} Speed:{}'.format(time_str, ii, jj, (time.time() - start)))
