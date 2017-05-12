'''Pre-activated Resnet for cifar dataset. 
Ported form https://github.com/facebook/fb.resnet.torch/blob/master/models/resattnet.lua
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F
import math
from .preresnet import BasicBlock, Bottleneck
from .hourglass import Hourglass


__all__ = ['Attention', 'ResAttNet', 'resattnet20', 'resattnet32', 'resattnet44', 'resattnet56',
           'resattnet110', 'resattnet1202']

class Attention(nn.Module):
    # implementation of Wang et al. "Residual Attention Network for Image Classification". CVPR, 2017.
    def __init__(self, block, p, t, r, planes, depth):
        super(Attention, self).__init__()
        self.p = p
        self.t = t
        out_planes = planes*block.expansion
        self.residual = block(out_planes, planes)
        self.hourglass = Hourglass(block, r, planes, depth)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        self.fc2 = nn.Conv2d(out_planes, 1, kernel_size=1, bias=False)

    def get_mask(self):
        return self.mx

    def forward(self, x):
        # preprocessing
        for i in range(0, self.p):
            x = self.residual(x)

        # trunk branch
        tx = x
        for i in range(0, self.p):
            tx = self.residual(tx)

        # mask branch
        mx = self.relu(self.bn(self.hourglass(x)))
        mx = self.fc1(mx)
        mx = self.fc2(mx)
        self.mx = F.sigmoid(mx)

        # residual attented feature
        out = tx + tx*self.mx.expand_as(tx)

        return out

class ResAttNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 16
        super(ResAttNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.att1 = Attention(block, 1, 2, 1, 16, 3)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.att2 = Attention(block, 1, 2, 1, 32, 2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.att3 = Attention(block, 1, 2, 1, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)
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
        masks.append(self.att1.get_mask())
        masks.append(self.att2.get_mask())
        return masks

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.att1(x)
        x = self.layer2(x)
        x = self.att2(x)
        x = self.layer3(x)
        x = self.att3(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resattnet20(**kwargs):
    """Constructs a ResAttNet-20 model.
    """
    model = ResAttNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resattnet32(**kwargs):
    """Constructs a ResAttNet-32 model.
    """
    model = ResAttNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resattnet44(**kwargs):
    """Constructs a ResAttNet-44 model.
    """
    model = ResAttNet(Bottleneck, [7, 7, 7], **kwargs)
    return model


def resattnet56(**kwargs):
    """Constructs a ResAttNet-56 model.
    """
    model = ResAttNet(Bottleneck, [9, 9, 9], **kwargs)
    return model


def resattnet110(**kwargs):
    """Constructs a ResAttNet-110 model.
    """
    model = ResAttNet(Bottleneck, [18, 18, 18], **kwargs)
    return model

def resattnet1202(**kwargs):
    """Constructs a ResAttNet-1202 model.
    """
    model = ResAttNet(Bottleneck, [200, 200, 200], **kwargs)
    return model