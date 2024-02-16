import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import DropPath

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

        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Block(nn.Module):

    def __init__(self, dim, layerscale_init_value, dp_rate):
        super(Block, self).__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, dim*4, kernel_size=1)
        self.act = QuickGELU()
        self.pwconv2 = nn.Conv2d(dim*4, dim, kernel_size=1)
        
        # layerscale
        self.gamma = nn.Parameter(layerscale_init_value * torch.ones((dim)), requires_grad=True) if layerscale_init_value > 0 else None
        # droppath(stochastic depth)
        self.droppath = DropPath(dp_rate) if dp_rate > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x *= self.gamma
        
        x = identity + self.droppath(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, 
                 block, 
                 dims=[96,192,384,768],
                 depths=[3,3,9,3], 
                 layerscale_init_value=1e-6,
                 droppath=0.1,
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
                                (f'ds_ln{i}', LayerNorm(dims[i+1])),
                                (f'ds_conv{i+1}', nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)),                                
                                ]))
            self.downsample_layers.append(downsample_layer)
        
        # stage layers
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, droppath, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dims[i], 
                        layerscale_init_value=layerscale_init_value,
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
    
def resnet50():
    return ResNet(Block, dims=[96,192,384,768], depths=[3, 3, 9, 3], num_classes=100)
