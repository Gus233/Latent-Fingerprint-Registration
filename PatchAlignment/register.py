
from __future__ import print_function
import os
import cv2
import torch
import time
import numpy as np
from scipy import io, misc
from matplotlib import  pyplot as plt

def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    return pairs


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None

   # image = image[np.newaxis, np.newaxis, :, :]
#    image = image.astype(np.float32, copy=False)
    return image



def lfw_test(pairs, test_root, test_list, theta_path, save_path):

    thetas = io.loadmat(theta_path)
    thetas = thetas['trans']
    cnt = 0
    for pair in pairs:

        data_path1 = os.path.join(test_root, pair.split()[0])
        data_path2 = os.path.join(test_root, pair.split()[1])
        image1 = load_image(data_path1)
        image2 = load_image(data_path2)


        theta = np.multiply(thetas[cnt], [80.0, 80.0, 180.0/np.pi])

        rows = 240
        cols = 240

        tmp = 255.0-image2
        M = np.float32([[1,0,theta[0]],[0,1,theta[1]]])
        tmp = cv2.warpAffine(tmp, M, (cols, rows))


        # plt.imshow(tmp)
        # plt.show()
        # cv2.imshow('img', tmp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        M = cv2.getRotationMatrix2D((cols/2, rows/2),theta[2],1.0)
        tmp = cv2.warpAffine(tmp, M, (cols, rows))
        tmp = 255.0-tmp
        cv2.imwrite(os.path.join(save_path,  pair.split()[0]), image1)
        cv2.imwrite(os.path.join(save_path, pair.split()[1][:-4] +'_' + pair.split()[0][:-4] + '.jpg'), tmp)



        cnt = cnt + 1
        if cnt % 1000 == 0:
            message = '[{}/{} ({:.0f}%)]'.format(
                cnt, len(pairs), 100. * cnt/ len(pairs))

            print(message)



if __name__ == '__main__':

    test_root = '/home/data2/gus/LatentMatch/nist27/fingernet_seg/test_iter_patch_pair_NIST27_fingernet_full/'
    test_list = '/home/data2/gus/LatentMatch/nist27/fingernet_seg/test_pair_iter_NIST27_fingernet_full.txt'
    theta_path = '/home/data2/gus/LatentMatch/nist27/fingernet_seg/test_pair_iter_NIST27_fingernet_full.mat'
    save_path = '/home/data2/gus/LatentMatch/nist27/fingernet_seg/test_iter_patch_pair_NIST27_fingernet_full_register/'

    pairs = get_lfw_list(test_list)
    lfw_test(pairs, test_root, test_list, theta_path, save_path)