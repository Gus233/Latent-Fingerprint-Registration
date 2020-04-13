


from __future__ import print_function
import os
import cv2
import torch
import time
import numpy as np
from scipy import io
from torch.nn import DataParallel

from resnet import *
from metrics import *
from focal_loss import *
from config import Config


def get_finger_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    return pairs


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = image[np.newaxis, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(finger_root, model, pairs, batch_size=10):

    transs = None
    cnt = 0
    for pair in pairs:
        data_path1 = os.path.join(finger_root, pair.split()[0])
        data_path2 = os.path.join(finger_root, pair.split()[1])
        image1 = load_image(data_path1)
        image2 = load_image(data_path2)

        data = (torch.from_numpy(image1), torch.from_numpy(image2))
        data = tuple(d.cuda() for d in data)

        output = model(*data)
        trans = output[0].data.cpu().numpy()

        if transs is None:
            transs = trans
        else:
            transs = np.concatenate((transs, trans), axis=0)

        cnt = cnt + 1
        if cnt % 1000 == 0:
            message = '[{}/{} ({:.0f}%)]'.format(
                cnt, len(pairs), 100. * cnt/ len(pairs))

            print(message)
    return transs


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)



def finger_test(finger_root, model, pairs,  batch_size):
    transs = get_featurs(finger_root, model, pairs, batch_size=batch_size)
    io.savemat(opt.save_path, {'trans': transs})



if __name__ == '__main__':

    opt = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    embedding_net = Unet_down()
    regression_net = Unet_up()
    model = AlignmentNet(embedding_net, regression_net)

    model = DataParallel(model)
    load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))

    pairs = get_finger_list(opt.finger_test_list)
    model.eval()
    finger_test(opt.finger_root, model, pairs,  opt.test_batch_size)
