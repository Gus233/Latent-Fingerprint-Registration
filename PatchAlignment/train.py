from __future__ import print_function
import os
import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
import random
import time
import numpy as np

from models import *
from lossfunction import *
from dataset import Dataset

from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from test import *





def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    opt = Config()

    device = torch.device("cuda")

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


    train_dataset = Dataset(opt.train_root, opt.pd_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    print('{} train iters per epoch:'.format(len(trainloader)))

    test_dataset = Dataset(opt.test_root, opt.test_pd_root, opt.test_list, phase='test', input_shape=opt.input_shape)
    testloader = data.DataLoader(test_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers)
    criterion = LossFunction()

    embedding_net=Unet_down()
    regression_net=Unet_up()

    model = AlignmentNet(embedding_net, regression_net)

    if opt.finetune:
        model = DataParallel(model)
        load_model(model, opt.load_model_path)
        model.load_state_dict(torch.load(opt.load_model_path))
        model.to(torch.device("cuda"))
    else:
        model.to(device)
        model = DataParallel(model)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)



    start = time.time()

    for i in range(opt.max_epoch):
        scheduler.step()
        model.train()
        losses = []
        losses1 = []
        losses2 = []
        losses3 = []
        total_loss = 0

        for ii, (data1, data2,  pddata, target) in enumerate(trainloader):
            target = target if len(target) > 0 else None

            data1 = data1.cuda()
            data2 = data2.cuda()
            if not type(pddata) in (tuple, list):
                pddata = (pddata,)
            pddata = tuple(d.cuda() for d in pddata)
            if target is not None:
                 target = target.cuda()
            if not type(target) in (tuple, list):
                target = (target,)

            optimizer.zero_grad()
            outputs = model(data1, data2)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            loss_inputs += pddata
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = criterion(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            losses1.append(loss_outputs[1].item())
            losses2.append(loss_outputs[2].item())
            losses3.append(loss_outputs[3].item())

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if ii % opt.print_freq == 0:
                message = 'Epoch: {}  Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Loss1: {:.6f} Loss2: {:.6f} Loss3: {:.6f}'.format(i,
                    ii * len(data1), len(trainloader.dataset),
                    100. * ii / len(trainloader), np.mean(losses),np.mean(losses1),np.mean(losses2),np.mean(losses3))


                print(message)
                losses = []
                losses1 = []
                losses2 = []
                losses3 = []

            if i % opt.save_interval == 0 or i == opt.max_epoch:
                save_model(model, opt.checkpoints_path, opt.backbone, i)


          

