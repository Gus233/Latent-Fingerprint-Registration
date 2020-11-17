
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)

class Unet_down(nn.Module):

    def __init__(self):
        super(Unet_down, self).__init__()
        self.conv1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)

        self.dropout = nn.Dropout()

        self.up6 = nn.ConvTranspose2d(256,128, 2, stride=2)
        self.conv6 = DoubleConv(128,128)
        self.conv10 = nn.Conv2d(128,2,1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        z = x
        x = self.conv4(x)

        z = self.up6(z)
        z = self.conv6(z)
        z = self.conv10(z)
        z = nn.Tanh()(z)
        return z, x


class Unet_up(nn.Module):
    def __init__(self):
        super(Unet_up, self).__init__()
        self.conv1 = DoubleConv(625, 256)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(256, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(128*6*6, 3)



    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
#        print(feature.size())
#        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class AlignmentNet(nn.Module):
    def __init__(self, embedding_net, regression_net):
        super(AlignmentNet, self).__init__()
        self.embedding_net = Unet_down()
        self.regreesion_net = Unet_up()

        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()
        self.ReLU = nn.ReLU()
    def forward(self, x1, x2):
        pd1,feature1 = self.embedding_net(x1)
        pd2,feature2 = self.embedding_net(x2)

        feature1 = self.FeatureL2Norm(feature1)
        feature2 = self.FeatureL2Norm(feature2)
        correlation = self.FeatureCorrelation(feature1, feature2)

        correlation = self.FeatureL2Norm(self.ReLU(correlation))
        trans = self.regreesion_net(correlation)

        return trans, pd1, pd2


    def get_embedding(self, x):
        pd, feature = self.embedding_net(x)
        return feature


