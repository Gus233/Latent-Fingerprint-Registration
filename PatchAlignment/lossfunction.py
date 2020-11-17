
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(LossFunction, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.L1Loss()#
        self.ce2 = torch.nn.MSELoss()

    def forward(self, input, output1, output2, target1, target2, label):
        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)
        target1 = target1.view(target1.size(0), -1)
        target2 = target2.view(target2.size(0), -1)


        loss1 = self.ce2(output1, target1)
        loss2 = self.ce2(output2, target2)

        loss3 = self.ce2(input, label[0])

        loss =  0.25*(loss1 + loss2) + loss3
        return (loss, loss1, loss2, loss3)

