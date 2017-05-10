'''
Hourglass network inserted in the pre-activated Resnet 
Use lr=0.01 for current version
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F
import math
from .preresnet import BasicBlock, Bottleneck

__all__ = ['Hourglass', 'HourglassNet', 'hgnet20', 'hgnet32', 'hgnet44', 'hgnet56',
           'hgnet110', 'hgnet1202']


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.residual = self._make_layer(block, num_blocks, planes)
        self.bn = nn.BatchNorm2d(planes)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def _make_layer(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _hour_glass(self, n, x):
        up1 = self.residual(x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.residual(low1)

        if n > 1:
            low2 = self._hour_glass(n-1, low1)
        else:
            low2 = self.residual(low1)
        low3 = self.residual(low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass(self.depth, x)

class HourglassNet(nn.Module):

    def __init__(self, block, layers, num_blocks=1, num_classes=1000):
        self.inplanes = 16
        super(HourglassNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], num_blocks=num_blocks, depth=3)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, num_blocks=num_blocks, depth=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, num_blocks=num_blocks, depth=1)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, num_blocks=1, depth=1):
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

        # add hourglass module
        layers.append(Hourglass(block, num_blocks, planes, depth))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def hgnet20(**kwargs):
    """Constructs a HourglassNet-20 model.
    """
    model = HourglassNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def hgnet32(**kwargs):
    """Constructs a HourglassNet-32 model.
    """
    model = HourglassNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def hgnet44(**kwargs):
    """Constructs a HourglassNet-44 model.
    """
    model = HourglassNet(Bottleneck, [7, 7, 7], **kwargs)
    return model


def hgnet56(**kwargs):
    """Constructs a HourglassNet-56 model.
    """
    model = HourglassNet(Bottleneck, [9, 9, 9], **kwargs)
    return model


def hgnet110(**kwargs):
    """Constructs a HourglassNet-110 model.
    """
    model = HourglassNet(Bottleneck, [18, 18, 18], **kwargs)
    return model

def hgnet1202(**kwargs):
    """Constructs a HourglassNet-1202 model.
    """
    model = HourglassNet(Bottleneck, [200, 200, 200], **kwargs)
    return model