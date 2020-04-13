
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

    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image



def cut_image(img, posx, posy, width):
    img = torch.unsqueeze(img,0)
    img = torch.unsqueeze(img, 0)

    posx = int(posx) + 120
    posy = int(posy) + 120
    image = img[:,:,posy-width:posy+width, posx-width:posx+width]

    return image

def get_featurs(I1, I2,  pairs,  register_model, similarity_model):

    batch_size = 20


    transs = None
    distances = None
    images1 = None
 #   images2 = None
    oimages2 = None
    cnt = 0


    I1 = np.pad(I1,((120,120),(120, 120)), 'constant', constant_values = 1.0)
    I2 = np.pad(I2, ((120, 120), (120, 120)), 'constant', constant_values=1.0)
    I1 = torch.from_numpy(I1).cuda()
    I2 = torch.from_numpy(I2).cuda()


    for pair in pairs:

        cnt = cnt + 1

        image1 = cut_image(I1, pair.split()[0], pair.split()[1], 80)
        oimage2 = cut_image(I2, pair.split()[2], pair.split()[3], 120)

        if images1 is None:
            images1 = image1
        #    images2 = image2
            oimages2 = oimage2
        else:
            images1 = torch.cat((images1, image1), 0)
        #    images2 = torch.cat((images2, image2), 0)
            oimages2 = torch.cat((oimages2, oimage2),0)

        images2 = oimages2[:,:,40:200,40:200]



        del image1, oimage2


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
            alpha = para[: ,2] /3 #
            theta =  torch.cat((para, para), 1)
            theta[: ,2] = -para[: ,0 ] *20.0/120.0
            theta[: ,5] = -para[: ,1 ] *20.0/120.0
            theta[: ,0] = torch.cos(alpha)
            theta[: ,1] = -torch.sin(alpha)
            theta[: ,3] = torch.sin(alpha)
            theta[: ,4] = torch.cos(alpha)
            theta = theta.reshape((para.shape[0], 2, 3))

            sampling_grid = AffineGridGen(240 ,240 ,1)(theta)
            warped_image_batch = F.grid_sample( 1.0 - oimages2,  sampling_grid)
            rec_images2 = (1.0-warped_image_batch)
            rec_images2 = rec_images2[: ,: ,40:200, 40:200]

            del para, theta, output,alpha
            del sampling_grid, warped_image_batch
            del images2, oimages2



          # --------  step 3   similarity -------------------------------  0.0017


            output1 = similarity_model.module.get_embedding(images1)
            output2 = similarity_model.module.get_embedding(rec_images2)

            distance = F.cosine_similarity(output1, output2)
            distance = distance.data

            del images1, rec_images2, output1, output2


            if distances is None:
                distances = distance
            else:
                distances = torch.cat((distances, distance), 0)

            images1 = None
          #  images2 = None
            oimages2 = None

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


def lfw_test(image1, image2, pairs,  register_model, similarity_model, trans_save_path,  sim_save_path):

    transs, sims, flag = get_featurs(image1, image2, pairs, register_model, similarity_model)

    if flag==1:
        io.savemat(trans_save_path, {'trans': transs})
        io.savemat(sim_save_path, {'sims': sims})


if __name__ == '__main__':
    register_model_path = '/home/data2/gus/LatentMatch/checkpoints/ave_20/select/resnet18_20.pth'   # threshold indeed 0.85'   #  select2
    # ave_40_wo_roi/combine2/resnet18_20.pth'   # select1'

    similarity_model_path = '/home/data2/gus/MinutiaeDescriptor/joint/checkpoints/create/32_new_160_add_texture_wo_roi/image_select80/resnet18_20.pth'  # select2'
    # 32_new_160_add_texture_wo_roi/resnet18_20.pth'  # select1



    # image_path = '/home/data2/gus/LatentMatch/nist27/gt_seg/all_pair/realcenter/test_pair_NIST27_image_register_reg_sim_select/'
    # minu_path = '/home/data2/gus/LatentMatch/nist27/gt_seg/all_pair/realcenter/test_iter_patch_pair_NIST27_list_reg_sim_select/'
    # trans_save_path = '/home/data2/gus/LatentMatch/nist27/gt_seg/all_pair/realcenter/test_iter_patch_pair_NIST27_result_reg_sim_select/'
    # sim_save_path =  trans_save_path

    image_path = '/home/data2/gus/MOLF/select_resize/'
    minu_path = '/home/data2/gus/LatentMatch/MOLF/test_patch_pair_list_dense/'
    trans_save_path = '/home/data2/gus/LatentMatch/MOLF/test_patch_result_dense/'
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


    for i in range(982,800,-1):
        for j in range(i,i+1):
            ii = i + 1

            jj = j + 1

            # data_path1 = os.path.join(image_path + str(ii), str(ii) + '_' + str(jj) + '_1.jpg')
            # image1 = load_image(data_path1)
            # data_path2 = os.path.join(image_path + str(ii), str(ii) + '_' + str(jj) + '_2.jpg')
            # image2 = load_image(data_path2)

            data_path1 = os.path.join(image_path + 'slap/', str(ii) +'.jpg')
            image1 = load_image(data_path1)
            data_path2 = os.path.join(image_path + 'latent/', str(jj) + '.jpg')
            image2 = load_image(data_path2)
            test_minu = os.path.join(minu_path + str(ii), str(ii) + '_' + str(jj) + '.txt')
            pairs = get_lfw_list(test_minu)

            if len(pairs) > 0:

                if not os.path.exists(trans_save_path + str(ii)):
                    os.makedirs(trans_save_path + str(ii))
                single_trans_save_path = os.path.join(trans_save_path + str(ii), str(ii) + '_' + str(jj) + '_trans.mat')
                single_sims_save_path = os.path.join(sim_save_path + str(ii), str(ii) + '_' + str(jj) + '_sims.mat')


                start = time.time()
                lfw_test(image1, image2,  pairs, register_model, similarity_model, single_trans_save_path, single_sims_save_path)
                time_str = time.asctime(time.localtime(time.time()))

                print('{} Current: {}_{} Speed:{}'.format(time_str, ii, jj, (time.time() - start) / len(pairs)))
