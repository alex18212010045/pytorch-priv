from __future__ import absolute_import

'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch.nn as nn
import math

import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from ..dropblock.dropblock import DropBlock2D
from ..dropblock.scheduler import LinearScheduler

__all__ = ['resdropnet18', 'resdropnet34', 'resdropnet50', 'resdropnet101', 'resdropnet152', 'resdropnet10_1x64d',
           'resdropnet18_1x64d', 'resdropnet50_1x64d', 'resdropnet101_1x64d']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ImageNetResNet(nn.Module):
    def __init__(self, bottleneck=False, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), drop_prob=0., block_size=5, num_classes=1000,nr_steps=5e3):
        """ Constructor
        Args:
            layers: config of layers, e.g., (3, 4, 23, 3)
            num_classes: number of classes
        """
        super(ResNet, self).__init__()
        if bottleneck:
            block = Bottleneck
        else:
            block = BasicBlock
        self.inplanes = baseWidth  # default 64

        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, baseWidth, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth)
        else:
            self.conv1 = nn.Conv2d(3, baseWidth // 2, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth // 2)
            self.conv2 = nn.Conv2d(baseWidth // 2, baseWidth // 2, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(baseWidth // 2)
            self.conv3 = nn.Conv2d(baseWidth // 2, baseWidth, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(baseWidth)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=nr_steps  # 5e3
        )
        self.layer1 = self._make_layer(block, baseWidth, layers[0])
        self.layer2 = self._make_layer(block, baseWidth * 2, layers[1], 2)
        self.layer3 = self._make_layer(block, baseWidth * 4, layers[2], 2)
        self.layer4 = self._make_layer(block, baseWidth * 8, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(baseWidth * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNet
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
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

    def forward(self, x):
        self.dropblock.step()
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropblock(self.layer3(x))
        x = self.dropblock(self.layer4(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
def resdropnet(**kwargs):
    """
    Construct ResNet.
    (2, 2, 2, 2) for resnet18	# bottleneck=False
    (2, 2, 2, 2) for resnet26
    (3, 4, 6, 3) for resnet34	# bottleneck=False
    (3, 4, 6, 3) for resnet50
    (3, 4, 23, 3) for resnet101
    (3, 8, 36, 3) for resnet152
    note: if you use head7x7=False, the actual depth of resnet will increase by 2 layers.
    """
    model = ImageNetResNet(**kwargs)
    return model


def resdropnet18(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=False, baseWidth=64, head7x7=True, layers=(2, 2, 2, 2), drop_prob=0., block_size=5, nr_steps=5e3, num_classes=1000)
    return model


def resdropnet34(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=False, baseWidth=64, head7x7=True, layers=(3, 4, 6, 3), num_classes=1000)
    return model


def resdropnet50(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 6, 3), drop_prob=0., block_size=5, nr_steps=5e3, num_classes=1000)
    return model


def resdropnet101(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), drop_prob=0., block_size=5, nr_steps=5e3, num_classes=1000)
    return model


def resdropnet152(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 8, 36, 3), drop_prob=0., block_size=5, nr_steps=5e3, num_classes=1000)
    return model


def resdropnet10_1x64d(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=False, baseWidth=64, head7x7=False, layers=(1, 1, 1, 1), drop_prob=0., block_size=5, nr_steps=5e3, num_classes=1000)
    return model


def resdropnet18_1x64d(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=False, baseWidth=64, head7x7=False, layers=(2, 2, 2, 2), drop_prob=0., block_size=5, nr_steps=5e3, num_classes=1000)
    return model


def resdropnet50_1x64d(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=True, baseWidth=64, head7x7=False, layers=(3, 4, 6, 3), drop_prob=0., block_size=5, nr_steps=5e3, num_classes=1000)
    return model


def resdropnet101_1x64d(drop_prob=0., block_size=5, nr_steps=5e3):
    model = ImageNetResNet(bottleneck=True, baseWidth=64, head7x7=False, layers=(3, 4, 23, 3), drop_prob=0., block_size=5, nr_steps=5e3, num_classes=1000)
    return model
