'''Pre-activated Resnet for cifar dataset. 
Ported form https://github.com/facebook/fb.resnet.torch/blob/master/models/resadvnet.lua
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F
import math
from .preresnet import BasicBlock, Bottleneck
from .hourglass import Hourglass
from .modules import SoftmaxAttention

# __all__ = ['Attention', 'ResAdvNet', 'resadvnet20', 'resadvnet32', 'resadvnet44', 'resadvnet56',
#            'resadvnet110', 'resadvnet1202']

# class Attention(nn.Module):
#     # implementation of Wang et al. "Residual Attention Network for Image Classification". CVPR, 2017.
#     def __init__(self, block, p, t, r, planes, depth):
#         super(Attention, self).__init__()
#         self.p = p
#         self.t = t
#         out_planes = planes*block.expansion
#         self.residual = block(out_planes, planes)
#         self.hourglass = Hourglass(block, r, planes, depth)
#         self.fc1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
#         self.fc2 = nn.Conv2d(out_planes, 1, kernel_size=1, bias=False)

#     def get_mask(self):
#         return self.mx

#     def forward(self, x):
#         # preprocessing
#         for i in range(0, self.p):
#             x = self.residual(x)

#         # trunk branch
#         tx = x
#         for i in range(0, self.p):
#             tx = self.residual(tx)

#         # mask branch
#         self.mx = F.sigmoid(self.fc2(self.fc1(self.hourglass(x))))

#         # residual attented feature
#         out = tx + tx*self.mx.expand_as(tx)

#         return out

class StackedAdversary(nn.Module):
    def __init__(self, block, planes, num_stacks=3, residual=False, normalize=False):
        super(StackedAdversary, self).__init__()
        self.num_stacks = num_stacks
        attentions = []
        for s in range(0, self.num_stacks):
            attentions.append(SoftmaxAttention(planes, normalize=normalize, residual=residual))
        self.attention = nn.ModuleList(attentions)

    def get_mask(self):
        mask = []
        for _, att in enumerate(self.attention):
            mask.append(att.get_mask())
        return mask

    def forward(self, x):
        out = x.clone()
        self.mask = []
        att_f = self.attention[0](x)
        out += att_f
        for s in range(1, self.num_stacks):
            adv_f = x - att_f
            att_f = self.attention[s](adv_f)
            x = adv_f
            out += att_f

        return out

class ResAdvNet(nn.Module):

    def __init__(self, block, layers, residual=False, normalize=False, num_classes=1000):
        self.inplanes = 16
        super(ResAdvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.adv1 = StackedAdversary(block, 16 * block.expansion, 
                            num_stacks=5, residual=False, normalize=False)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.adv2 = StackedAdversary(block, 32 * block.expansion, 
                            num_stacks=5, residual=False, normalize=False)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.adv3 = StackedAdversary(block, 64 * block.expansion, 
                            num_stacks=5, residual=False, normalize=False)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        # masks.append(self.adv1.get_mask())
        masks.append(self.adv2.get_mask())
        masks.append(self.adv3.get_mask())
        return masks

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        # x = self.adv1(x)
        x = self.layer2(x)
        x = self.adv2(x)
        x = self.layer3(x)
        x = self.adv3(x)
        x = self.relu(self.bn(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resadvnet20(**kwargs):
    """Constructs a ResAdvNet-20 model.
    """
    model = ResAdvNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resadvnet32(**kwargs):
    """Constructs a ResAdvNet-32 model.
    """
    model = ResAdvNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resadvnet44(**kwargs):
    """Constructs a ResAdvNet-44 model.
    """
    model = ResAdvNet(Bottleneck, [7, 7, 7], **kwargs)
    return model


def resadvnet56(**kwargs):
    """Constructs a ResAdvNet-56 model.
    """
    model = ResAdvNet(Bottleneck, [9, 9, 9], **kwargs)
    return model


def resadvnet110(**kwargs):
    """Constructs a ResAdvNet-110 model.
    """
    model = ResAdvNet(Bottleneck, [18, 18, 18], **kwargs)
    return model

def resadvnet1202(**kwargs):
    """Constructs a ResAdvNet-1202 model.
    """
    model = ResAdvNet(Bottleneck, [200, 200, 200], **kwargs)
    return model

# ----------------------------

def resadvbn20(**kwargs):
    """Constructs a ResAdvNet-20 model.
    """
    model = ResAdvNet(BasicBlock, [3, 3, 3], normalize=True, **kwargs)
    return model


def resadvbn32(**kwargs):
    """Constructs a ResAdvNet-32 model.
    """
    model = ResAdvNet(BasicBlock, [5, 5, 5], normalize=True, **kwargs)
    return model


def resadvbn44(**kwargs):
    """Constructs a ResAdvNet-44 model.
    """
    model = ResAdvNet(Bottleneck, [7, 7, 7], normalize=True, **kwargs)
    return model


def resadvbn56(**kwargs):
    """Constructs a ResAdvNet-56 model.
    """
    model = ResAdvNet(Bottleneck, [9, 9, 9], normalize=True, **kwargs)
    return model


def resadvbn110(**kwargs):
    """Constructs a ResAdvNet-110 model.
    """
    model = ResAdvNet(Bottleneck, [18, 18, 18], normalize=True, **kwargs)
    return model

def resadvbn1202(**kwargs):
    """Constructs a ResAdvNet-1202 model.
    """
    model = ResAdvNet(Bottleneck, [200, 200, 200], normalize=True, **kwargs)
    return model

# -------------------------------------

def resadvbnres20(**kwargs):
    """Constructs a ResAdvNet-20 model.
    """
    model = ResAdvNet(BasicBlock, [3, 3, 3], normalize=True, residual=True, **kwargs)
    return model


def resadvbnres32(**kwargs):
    """Constructs a ResAdvNet-32 model.
    """
    model = ResAdvNet(BasicBlock, [5, 5, 5], normalize=True, residual=True, **kwargs)
    return model


def resadvbnres44(**kwargs):
    """Constructs a ResAdvNet-44 model.
    """
    model = ResAdvNet(Bottleneck, [7, 7, 7], normalize=True, residual=True, **kwargs)
    return model


def resadvbnres56(**kwargs):
    """Constructs a ResAdvNet-56 model.
    """
    model = ResAdvNet(Bottleneck, [9, 9, 9], normalize=True, residual=True, **kwargs)
    return model


def resadvbnres110(**kwargs):
    """Constructs a ResAdvNet-110 model.
    """
    model = ResAdvNet(Bottleneck, [18, 18, 18], normalize=True, residual=True, **kwargs)
    return model

def resadvbnres1202(**kwargs):
    """Constructs a ResAdvNet-1202 model.
    """
    model = ResAdvNet(Bottleneck, [200, 200, 200], normalize=True, residual=True, **kwargs)
    return model