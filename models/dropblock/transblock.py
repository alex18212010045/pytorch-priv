import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
class transBlocklayer3(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

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

    def __init__(self, drop_prob, block_size, premodel,layer3):
        super(transBlocklayer3, self).__init__()
        # modified by Riheng 2018/12/06
        self.drop_prob = drop_prob
        #self.register_buffer('drop_prob', torch.FloatTensor([drop_prob]))
        self.block_size = block_size
        self.premodel=premodel
        self.layer3 = layer3

    
    def forward(self, x):
        # shape: (bsize, channels, height, width)
        #2018/12/7
        #record the outputs
        #check the architecture of premodel
        output4=self.premodel.layer3(x)

        x=self.layer3(x)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

 
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            
            #2018/12/7
            #transfer the output
            block_mask_np=block_mask.numpy()
            x=x.numpy()
            out=x
            
            for a in block_mask_np:
                m,n,j,k = a.shape
                for i in range(m):
                    for j in range(n):
                        for q in range(j):
                            for w in range(k):
                                if a[i,j]==1 :
                                    out[i,j,q,w]=x[i,j,q,w]
                                else:
                                    out[i,j,q,w]=output4[i,j,q,w]
                                        
            out=torch.from_numpy(out)
            
            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

class transBlocklayer4(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

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

    def __init__(self, drop_prob, block_size, premodel,layer4):
        super(transBlocklayer4, self).__init__()

        # modified by Riheng 2018/12/06
        self.drop_prob = drop_prob
        #self.register_buffer('drop_prob', torch.FloatTensor([drop_prob]))
        self.block_size = block_size
        self.premodel=premodel
        self.layer4=layer4
        
    def forward(self, x):
        # shape: (bsize, channels, height, width)
        
        #2018/12/7
        #record the outputs
        #check the architecture of premodel
        output4=self.premodel.layer4(x)
        x=self.layer4(x)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

 
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            
            #2018/12/7
            #transfer the output
            block_mask_np=block_mask.numpy()
            x=x.numpy()
            out=x
            
            for a in block_mask_np:
                m,n,j,k = a.shape
                for i in range(m):
                    for j in range(n):
                        for q in range(j):
                            for w in range(k):
                                if a[i,j]==1 :
                                    out[i,j,q,w]=x[i,j,q,w]
                                else:
                                    out[i,j,q,w]=output4[i,j,q,w]
                                        
            out=torch.from_numpy(out)
            
            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)