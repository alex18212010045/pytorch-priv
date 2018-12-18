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

__all__ = ['resdropnet20','resdropnet32','resdropnet44','resdropnet56','resdropnet110','resdropnet1202']

class ChannelPool(nn.Module):
    def __init__(self, kernel_size, stride, dilation=1, padding=0, pool_type='Max'):
        super(ChannelPool, self).__init__()
        if pool_type == 'Max':
            self.pool3d = nn.MaxPool3d((kernel_size,1,1),stride =(stride,1,1),padding = (padding,0,0),dilation = (dilation,1,1))
        elif pool_type == 'Avg':
            self.pool3d = AvgPool3d((stride,1,1),stride = (stride,1,1))
    def forward(self,x):
        n,c,h,w = x.size()
        x = x.view(n,1,c,h,w)
        y = self.pool3d(x)
        n,c,d,h,w = y.size()
        y = y.view(n,d,h,w)
        return y


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
        self.stride = stride

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_channel_pool=False):
        super(Bottleneck, self).__init__()
        self.use_channel_pool = use_channel_pool
        if self.use_channel_pool:  # stride=4, kernel=6, pad=1
            pool_stride = inplanes / planes
            pool_kernel = pool_stride + 2
            pool_padding = 1
            self.cp = ChannelPool(pool_kernel, stride=pool_stride, padding=pool_padding, pool_type='Max')
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        if self.use_channel_pool:
            out = self.cp(x)
        else:
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


class CifarResDropNet(nn.Module):

    def __init__(self, depth, num_classes=1000, drop_prob=0., block_size=5, nr_steps=5e3):
        super(CifarResDropNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) / 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=nr_steps  # 5e3
        )

        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
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

    def forward(self, x):
        self.dropblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16

        # x = self.layer3(x)  # 8x8
        x = self.dropblock(self.layer3(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resdropnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return CifarResDropNet(**kwargs)

def resdropnet20(drop_prob=0., block_size=5, nr_steps=5e3):
    model = CifarResDropNet(depth = 20, num_classes=10, drop_prob=drop_prob, block_size=block_size, nr_steps=nr_steps)
    return model

def resdropnet32(drop_prob=0., block_size=5, nr_steps=5e3):
    model = CifarResDropNet(depth = 32, num_classes=10, drop_prob=drop_prob, block_size=block_size, nr_steps=nr_steps)
    return model

def resdropnet44(drop_prob=0., block_size=5, nr_steps=5e3):
    model = CifarResDropNet(depth = 44, num_classes=10, drop_prob=drop_prob, block_size=block_size, nr_steps=nr_steps)
    return model

def resdropnet56(drop_prob=0., block_size=5, nr_steps=5e3):
    model = CifarResDropNet(depth = 56, num_classes=10, drop_prob=drop_prob, block_size=block_size, nr_steps=nr_steps)
    return model

def resdropnet110(drop_prob=0., block_size=5, nr_steps=5e3):
    model = CifarResDropNet(depth = 110, num_classes=10, drop_prob=drop_prob, block_size=block_size, nr_steps=nr_steps)
    return model

def resdropnet1202(drop_prob=0., block_size=5, nr_steps=5e3):
    model = CifarResDropNet(depth = 1202, num_classes=10, drop_prob=drop_prob, block_size=block_size, nr_steps=nr_steps)
    return model