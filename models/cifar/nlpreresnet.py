from __future__ import absolute_import

'''Preactivated Resnet with non-local operations for cifar dataset.
https://arxiv.org/abs/1711.07971

> Wang X, Girshick R, Gupta A, et al.
> Non-local Neural Networks.
> arXiv preprint arXiv:1711.07971, 2017.

(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['preresnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class NonLocalBlock(nn.Module):
    def __init__(self, planes, expansion=4):
        super(NonLocalBlock, self).__init__()
        self.hplanes = planes // expansion

        self.theta = nn.Conv2d(planes, self.hplanes, kernel_size=1)
        self.phi = nn.Conv2d(planes, self.hplanes, kernel_size=1)
        self.g = nn.Conv2d(planes, self.hplanes, kernel_size=1)
        self.z = nn.Conv2d(self.hplanes, planes, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.01
        self.theta.weight.data.normal_(mean=0, std=std)
        self.theta.bias.data.zero_()
        self.phi.weight.data.normal_(mean=0, std=std)
        self.phi.bias.data.zero_()
        self.g.weight.data.normal_(mean=0, std=std)
        self.g.bias.data.zero_()
        self.z.weight.data.zero_()
        self.z.bias.data.zero_()

    def forward(self, x):
        height, width = x.size(2), x.size(3)

        q_ = self.theta(x)
        q_ = q_.view(q_.size(0), q_.size(1), q_.size(2) * q_.size(3))

        xpool = F.max_pool2d(x, 2)
        k_ = self.phi(xpool)
        k_ = k_.view(k_.size(0), k_.size(1), k_.size(2) * k_.size(3))

        v_ = self.g(xpool)
        v_ = v_.view(v_.size(0), v_.size(1), v_.size(2) * v_.size(3))

        qk = torch.matmul(q_.permute(0, 2, 1), k_) * (self.hplanes ** -.5)
        p = F.softmax(qk, dim=2)
        vp = torch.matmul(v_, p.permute(0, 2, 1))
        vp = vp.view(vp.size(0), vp.size(1), height, width)
        return self.z(vp) + x


class PreResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(PreResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.nonlocal1 = NonLocalBlock(16)
        self.nonlocal2 = NonLocalBlock(16 * block.expansion)
        self.nonlocal3 = NonLocalBlock(32 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        # x = self.nonlocal1(x)
        x = self.layer1(x)  # 32x32
        # x = self.nonlocal2(x)
        x = self.layer2(x)  # 16x16
        x = self.nonlocal3(x)
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def nlpreresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return PreResNet(**kwargs)
