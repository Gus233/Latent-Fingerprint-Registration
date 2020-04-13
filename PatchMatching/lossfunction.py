# -*- coding: utf-8 -*-
"""
Created on 18-6-7 上午10:11

@author: ronghuaiyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        losses = loss.mean() + (loss1 + loss2)/2
        return (losses, loss1, loss2, loss.mean())

