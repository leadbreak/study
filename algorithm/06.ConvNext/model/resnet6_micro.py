import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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

class invertedBottleneck(nn.Module):
    expansion = 4  # 확장 비율

    def __init__(self, dim):
        super(invertedBottleneck, self).__init__()
        
        # move up depthwise conv with larger kernel
        self.block1 = nn.Sequential(OrderedDict([
            ('dwconv', nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)),
            ('layernorm', LayerNorm(dim)),
        ]))
        
        # first pointwise convolution(make wider)
        self.block2 = nn.Sequential(OrderedDict([
            ('pwconv1', nn.Conv2d(dim, dim*self.expansion, kernel_size=1)),
            ('activation', QuickGELU())
        ]))

        # pointwise convolution
        self.block3 = nn.Sequential(OrderedDict([
            ('pwconv2', nn.Conv2d(dim*self.expansion, dim, kernel_size=1)),
        ]))

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x += identity         
        
        return x

class ResNet(nn.Module):
    def __init__(self, 
                 block, 
                 dims=[96,192,384,768],
                 depths=[3,4,6,3], 
                 num_classes=100):
        super(ResNet, self).__init__()
        
        # Patchify Stem
        self.stem = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(3, dims[0], kernel_size=4, stride=4)),
            ('stem_ln', LayerNorm(dims[0])),
        ]))
        
        # downsample layers
        self.downsample_layers = nn.ModuleList()    
        self.downsample_layers.append(self.stem)    
        
        for i in range(3):
            downsample_layer = nn.Sequential(OrderedDict([
                                (f'ds_ln{i}', LayerNorm(dims[i])),
                                (f'ds_conv{i+1}', nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)),                                
                                ]))
            self.downsample_layers.append(downsample_layer)
        
        # stage layers
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[block(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

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
    
def resnet50():
    return ResNet(invertedBottleneck, dims=[96,192,384,768], depths=[3, 3, 9, 3], num_classes=100)
