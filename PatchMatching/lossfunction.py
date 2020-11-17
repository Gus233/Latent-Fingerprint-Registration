# -*- coding: utf-8 -*-
"""
Created on 18-6-7 上午10:11

@author: ronghuaiyang
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.l1 = nn.MSELoss()

    def forward(self, feature1, feature2, output1, output2, target1, target2, label, size_average=True):#, minu_output1, minu_output2, target1, target2, minu1, minu2, label, size_average=True):
        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)
        target1 = target1.view(target1.size(0), -1)
        target2 = target2.view(target2.size(0), -1)
        loss1 = self.l1(output1, target1)
        loss2 = self.l1(output2, target2)

        label = label[0]
        distances = (feature1 - feature2).pow(2).sum(1)  # squared distances
        losses = 0.5 * (label.float() * distances + (1 + -1 * label).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))


        loss = losses.mean()  + loss1 + loss2
        return (loss, loss1, loss2, losses.mean())

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

        self.l1 = nn.MSELoss()
    def forward(self, feature1, feature2, output1, output2, target1, target2, label):
        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)
        target1 = target1.view(target1.size(0), -1)
        target2 = target2.view(target2.size(0), -1)
        loss1 = self.l1(output1, target1)
        loss2 = self.l1(output2, target2)




        embeddings = torch.cat((feature1, feature2), dim = 0)
        target = torch.cat((label[0], label[1]), dim = 0)
        positive_pairs, negative_pairs  = self.pair_selector.get_pairs(embeddings, target)

        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
               self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)


        # positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        # negative_loss = F.relu(self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(1))  # revise 0429


        loss = torch.cat([positive_loss, negative_loss], dim=0)
        losses = loss.mean() + (loss1 + loss2)/2


        return (losses, loss1, loss2, loss.mean())



class OnlinePairLoss(nn.Module):


    def __init__(self,  pair_selector):
        super(OnlinePairLoss, self).__init__()
        self.pair_selector = pair_selector
        self.l1 = nn.MSELoss()
    def forward(self, feature1, feature2, output1, output2, target1, target2, label):
        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)
        target1 = target1.view(target1.size(0), -1)
        target2 = target2.view(target2.size(0), -1)
        loss1 = self.l1(output1, target1)
        loss2 = self.l1(output2, target2)




        embeddings = feature1
        target = label[0]
        positive_pairs, negative_pairs  = self.pair_selector.get_pairs(embeddings, target)


        if positive_pairs.shape[0] ==0:
            anchor = feature1
            positive = feature2
            negative = feature2.unsqueeze(1).expand(feature1.shape[0],feature1.shape[0]-1, 512)
        else:
            anchor = feature1[:positive_pairs[0,1]-1,:]
            positive = feature2[:positive_pairs[0,1]-1,:]
            negative = positive.unsqueeze(1).expand(positive_pairs[0,1]-1, positive_pairs[0,1]-2, 512)

        loss3 = torch.sum(anchor ** 2 + positive ** 2) / anchor.shape[0]


        anchor = anchor.unsqueeze(1)
        positive = positive.unsqueeze(1)
        x = torch.matmul(anchor, (negative-positive).transpose(1,2))
        x = torch.sum(torch.exp(x),2)
        loss = torch.mean(torch.log(1+x))


        losses = loss + (loss1 + loss2)/2 + 0.02*loss3

        return (losses, loss1, loss2, loss.mean())

