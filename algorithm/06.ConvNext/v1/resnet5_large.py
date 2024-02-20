import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class invertedBottleneck(nn.Module):
    expansion = 4  # 확장 비율

    def __init__(self, dim):
        super(invertedBottleneck, self).__init__()
        
        # move up depthwise conv with larger kernel
        self.block1 = nn.Sequential(OrderedDict([
            ('dwconv', nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)),
            ('bn1', nn.BatchNorm2d(dim)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        
        # first pointwise convolution(make wider)
        self.block2 = nn.Sequential(OrderedDict([
            ('pwconv1', nn.Conv2d(dim, dim*self.expansion, kernel_size=1)),
            ('bn2', nn.BatchNorm2d(dim*self.expansion)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

        # pointwise convolution
        self.block3 = nn.Sequential(OrderedDict([
            ('pwconv2', nn.Conv2d(dim*self.expansion, dim, kernel_size=1)),
            ('bn3', nn.BatchNorm2d(dim)),
        ]))
        
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x += identity         
        x = self.relu3(x)  
        
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
            ('stem_bn', nn.BatchNorm2d(dims[0])),
            ('stem_relu', nn.ReLU(inplace=True)),               
        ]))
        
        # downsample layers
        self.downsample_layers = nn.ModuleList()    
        self.downsample_layers.append(self.stem)    
        
        for i in range(3):
            downsample_layer = nn.Sequential(OrderedDict([
                                (f'ds_conv{i+1}', nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)),
                                (f'ds_bn{i+1}', nn.BatchNorm2d(dims[i+1])),
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
        self.fc = nn.Linear(dims[3], num_classes)
        
    def forward_features(self, x):
        for i in range(4):        
            x = self.downsample_layers[i](x)    
            x = self.stages[i](x)        
        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnet50():
    return ResNet(invertedBottleneck, dims=[96,192,384,768], depths=[3, 3, 9, 3], num_classes=100)
