'''
convnext v2 포스팅에서 구현한 모델
'''

import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import DropPath

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class LayerNorm(nn.Module):
    '''
    LayerNormalization for Channel First
    This is same with nn.LayerNorm(specialized for nn.Linear - Channel Last) after reshape    
    '''

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True, unbiased=False).sqrt()
        x = (x - mean) / (std + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

class Block(nn.Module):

    def __init__(self, dim, dp_rate):
        super(Block, self).__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim*4)
        self.act = QuickGELU()
        self.grn = GRN(dim*4) # Global Response Normalization
        self.pwconv2 = nn.Linear(dim*4, dim)
        
        # droppath(stochastic depth)
        self.droppath = DropPath(dp_rate) if dp_rate > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1) # (N, C, H, W) -> (N, H, W, C) : For Channel-wise norm
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0,3,1,2) # (N, H, W, C) -> (N, C, H, W)
        
        x = identity + self.droppath(x)
        
        return x

class ConvNeXtV2(nn.Module):
    def __init__(self,  
                 in_chans=3,
                 dims=[96,192,384,768],
                 depths=[3,3,9,3], 
                 droppath=0.1,
                 num_classes=100):
        super(ConvNeXtV2, self).__init__()
        
        # Patchify Stem
        stem = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)),
            ('stem_ln', LayerNorm(dims[0])),
        ]))
        
        # downsample layers
        self.downsample_layers = nn.ModuleList()    
        self.downsample_layers.append(stem)    
        
        for i in range(3):
            downsample_layer = nn.Sequential(OrderedDict([
                                (f'ds_ln', LayerNorm(dims[i])),
                                (f'ds_conv', nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)),                                
                                ]))
            self.downsample_layers.append(downsample_layer)
        
        # stage layers
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, droppath, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dims[i], 
                        dp_rate=dp_rates[cur+j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            
            cur += depths[i]

        # 평균 풀링과 Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layernorm = nn.LayerNorm(dims[3])      # Channel Last Layernorm
        self.fc = nn.Linear(dims[3], num_classes)
        
    def forward_features(self, x):
        for i in range(4):        
            x = self.downsample_layers[i](x)    
            x = self.stages[i](x)        
        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.avgpool(x)     # (N, C, H, W) -> (N, C, 1, 1)
        x = torch.flatten(x, 1) # (N, C, 1, 1) -> (N, C)
        x = self.layernorm(x)
        x = self.fc(x)
        return x
    
def load_convNext(dims=[96,192,384,768], depths=[3, 3, 9, 3], num_classes=100, **args):
    return ConvNeXtV2(dims=dims, depths=depths, num_classes=num_classes, **args)
