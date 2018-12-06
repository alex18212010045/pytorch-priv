from __future__ import division

"""
Creates a ResNet Model as defined in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2015). 
Deep Residual Learning for Image Recognition. 
arXiv preprint arXiv:1512.03385.
import from https://github.com/facebook/fb.resnet.torch
Copyright (c) Yang Lu, 2017
"""
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from dropblock import LinearScheduler
from torch.distributions import Bernoulli
import torchvision
from torch.autograd import Variable

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet10_1x64d',
           'resnet18_1x64d', 'resnet50_1x64d', 'resnet101_1x64d']


class DropBlock2D(nn.Module):
    
    """Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        

        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:        
            time=0
            model = torchvision.models.resnet50(pretrained=True)
            output1 = x
            output1=model.conv1(output1)
            output1=model.bn1(output1)
            output1=model.relu(output1)
            output1=model.maxpool(output1)
            output1=model.layer1(output1)
            output1=model.layer2(output1)
            output1=model.layer3(output1)
            output2=model.layer4(output1)
            # sample from a mask
            mask_reduction = self.block_size // 2
            mask_height = x.shape[-2] - mask_reduction
            mask_width = x.shape[-1] - mask_reduction
            mask_sizes = [mask_height, mask_width]

            if any([x <= 0 for x in mask_sizes]):
                raise ValueError('Input of shape {} is too small for block_size {}'
                                 .format(tuple(x.shape), self.block_size))

            # get gamma value
            gamma = self._compute_gamma(x, mask_sizes)
            # sample mask
            mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes))
            # place mask on input device
            mask = mask.to(x.device)
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            print(block_mask)
            print(block_mask[:,None,:,:])
            # apply block mask
            block_mask_np=block_mask.numpy()
            x=x.numpy()
            out=x
            print(out.shape)
            for a in block_mask_np:
                m,n,j,k = a.shape
                for i in range(m):
                    for j in range(n):
                        for q in range(j):
                            for w in range(k):
                                if a[i,j]==1 :
                                    out[i,j,q,w]=x[i,j,q,w]
                                else :
                                    if time ==0 :
                                        out[i,j,q,w]=output1[i,j,q,w]
                                        time=1
                                    else :
                                        out[i,j,q,w]=output2[i,j,q,w]
                                        time=0
            out=torch.from_numpy(out)
            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.conv2d(mask[:, None, :, :],
                              torch.ones((1, 1, self.block_size, self.block_size)).to(
                                  mask.device),
                              padding=int(np.ceil(self.block_size / 2) + 1))
        delta = self.block_size // 2
        input_height = mask.shape[-2] + delta
        input_width = mask.shape[-1] + delta

        height_to_crop = block_mask.shape[-2] - input_height
        width_to_crop = block_mask.shape[-1] - input_width

        if height_to_crop != 0:
            block_mask = block_mask[:, :, :-height_to_crop, :]

        if width_to_crop != 0:
            block_mask = block_mask[:, :, :, :-width_to_crop]
        block_mask = (block_mask >= 1).to(device=block_mask.device, dtype=block_mask.dtype)
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x, mask_sizes):
        feat_area = x.shape[-2] * x.shape[-1]
        mask_area = mask_sizes[-2] * mask_sizes[-1]
        return (self.drop_prob / (self.block_size ** 2)) * (feat_area / mask_area)
    
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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


class ResNet(nn.Module):
    def __init__(self, bottleneck=False, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000, drop_prob=0., block_size=5):
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
        self.dropblock = LinearScheduler(DropBlock2D(0.,5),start_value=0.,stop_value=drop_prob,nr_steps=5e3)
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


def resnet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000):
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
    model = ResNet(bottleneck=bottleneck, baseWidth=baseWidth, head7x7=head7x7, layers=layers, num_classes=num_classes)
    return model


def resnet18():
    model = ResNet(bottleneck=False, baseWidth=64, head7x7=True, layers=(2, 2, 2, 2), num_classes=1000)
    return model


def resnet34():
    model = ResNet(bottleneck=False, baseWidth=64, head7x7=True, layers=(3, 4, 6, 3), num_classes=1000)
    return model


def resnet50():
    model = ResNet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 6, 3), num_classes=1000)
    return model


def resnet101():
    model = ResNet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000)
    return model


def resnet152():
    model = ResNet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 8, 36, 3), num_classes=1000)
    return model


def resnet10_1x64d():
    model = ResNet(bottleneck=False, baseWidth=64, head7x7=False, layers=(1, 1, 1, 1), num_classes=1000)
    return model


def resnet18_1x64d(drop_prob,block_size):
    model = ResNet(drop_prob=drop_prob, block_size=block_size, bottleneck=False, baseWidth=64, head7x7=False, layers=(2, 2, 2, 2), num_classes=1000)
    return model


def resnet50_1x64d(drop_prob,block_size):
    model = ResNet(drop_prob=drop_prob, block_size=block_size, bottleneck=True, baseWidth=64, head7x7=False, layers=(3, 4, 6, 3), num_classes=1000)
    return model


def resnet101_1x64d():
    model = ResNet(bottleneck=True, baseWidth=64, head7x7=False, layers=(3, 4, 23, 3), num_classes=1000)
    return model
