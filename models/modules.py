'''
Useful modules for building deep neural networks
Copyright (c) YANG, Wei 2017
'''
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['SoftmaxAttention', 'SigmoidAttention']

class SoftmaxAttention(nn.Module):
    # implementation of Wang et al. "Residual Attention Network for Image Classification". CVPR, 2017.
    def __init__(self, planes, residual=True):
        super(SoftmaxAttention, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax()
        self.mask = None

    def get_mask(self):
        return self.mask

    def forward(self, x):
        mask = self.conv1(x)
        mask = self.bn(mask)
        mask = self.relu(mask)
        mask = self.conv2(mask)
        mask = mask.view(mask.size(0), -1)
        mask = self.softmax(mask)
        mask = mask.view(mask.size(0), 1, x.size(2), x.size(3))
        const = F.max_pool2d(mask, mask.size(2))
        mask = mask / const.expand_as(mask)
        self.mask = mask


        out = x * mask.expand_as(x)
        if self.residual:
            out += x

        return out


class SigmoidAttention(nn.Module):
    # implementation of Wang et al. "Residual Attention Network for Image Classification". CVPR, 2017.
    def __init__(self, planes, residual=True):
        super(SigmoidAttention, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax()
        self.mask = None

    def get_mask(self):
        return self.mask

    def forward(self, x):
        mask = self.conv1(x)
        mask = self.bn(mask)
        mask = self.relu(mask)
        mask = self.conv2(mask)
        mask = self.sigmoid(mask)
        self.mask = mask


        out = x * mask.expand_as(x)
        if self.residual:
            out += x

        return out

# model = SoftmaxAttention(3).cuda()
# x = torch.randn(2,3,32,32)
# out = model(Variable(x.cuda()))
# print(out.size())