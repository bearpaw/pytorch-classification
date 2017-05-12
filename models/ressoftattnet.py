'''Pre-activated Resnet for cifar dataset. 
Ported form https://github.com/facebook/fb.resnet.torch/blob/master/models/ressoftattnet.lua
(c) YANG, Wei 

V2: softattention
'''
import torch.nn as nn
import torch.nn.functional as F
import math
from .preresnet import BasicBlock, Bottleneck
from .hourglass import Hourglass
from .modules import *


__all__ = ['ResSoftAttNet', 'ressoftattnet20', 'ressoftattnet32', 'ressoftattnet44', 'ressoftattnet56',
           'ressoftattnet110', 'ressoftattnet1202']

class ResSoftAttNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 16
        super(ResSoftAttNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.att = SoftmaxAttention(64)
        self.bn = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_mask(self): # get attention mask
        masks = []
        masks.append(self.att.get_mask())
        return masks

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.att(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ressoftattnet20(**kwargs):
    """Constructs a ResSoftAttNet-20 model.
    """
    model = ResSoftAttNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def ressoftattnet32(**kwargs):
    """Constructs a ResSoftAttNet-32 model.
    """
    model = ResSoftAttNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def ressoftattnet44(**kwargs):
    """Constructs a ResSoftAttNet-44 model.
    """
    model = ResSoftAttNet(Bottleneck, [7, 7, 7], **kwargs)
    return model


def ressoftattnet56(**kwargs):
    """Constructs a ResSoftAttNet-56 model.
    """
    model = ResSoftAttNet(Bottleneck, [9, 9, 9], **kwargs)
    return model


def ressoftattnet110(**kwargs):
    """Constructs a ResSoftAttNet-110 model.
    """
    model = ResSoftAttNet(Bottleneck, [18, 18, 18], **kwargs)
    return model

def ressoftattnet1202(**kwargs):
    """Constructs a ResSoftAttNet-1202 model.
    """
    model = ResSoftAttNet(Bottleneck, [200, 200, 200], **kwargs)
    return model