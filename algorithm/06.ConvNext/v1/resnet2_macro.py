import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Bottleneck(nn.Module):
    expansion = 4  # 확장 비율, Bottleneck 구조에서 마지막 Conv Layer의 출력 채널이 입력 채널의 4배가 됨

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        
        # 첫 번째 1x1 Convolution
        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        
        # 두 번째 3x3 Convolution, stride는 주어진 값 사용
        self.block2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

        # 세 번째 1x1 Convolution, 채널 확장
        self.block3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)),
            ('bn3', nn.BatchNorm2d(out_channels * self.expansion)),
        ]))
        
        self.shortcut = shortcut
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
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
        
        self.in_channels = dims[0]
        
        # Patchify Stem
        self.stem = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, dims[0], kernel_size=4, stride=4)),
            ('bn', nn.BatchNorm2d(dims[0])),
            ('relu', nn.ReLU(inplace=True)),               
        ]))
        
        # shortcut layers for residual
        self.shortcut_layers = nn.ModuleList()        
        
        for i in range(4):
            if i == 0:
                shortcut_layer = nn.Sequential(OrderedDict([
                                    (f'ds_conv{i+1}', nn.Conv2d(dims[i], dims[i]*4, kernel_size=1, stride=1, bias=False)),
                                    (f'ds_bn{i+1}', nn.BatchNorm2d(dims[i]*4)),
                                    ]))
            else :
                shortcut_layer = nn.Sequential(OrderedDict([
                                    (f'ds_conv{i+1}', nn.Conv2d(dims[i]*2, dims[i]*4, kernel_size=1, stride=2, bias=False)),
                                    (f'ds_bn{i+1}', nn.BatchNorm2d(dims[i]*4)),
                                    ]))
            self.shortcut_layers.append(shortcut_layer)
        
        # stage layers
        self.stages = nn.ModuleList()
        for idx, dim in enumerate(dims):
            stages = []
            stride = 1 if idx == 0 else 2
            stages.append(block(self.in_channels, dim, stride=stride, shortcut=self.shortcut_layers[idx]))
            self.in_channels = dim * block.expansion
            for _ in range(1, depths[idx]):
                stages.append(block(self.in_channels, dim, shortcut=None))
            self.stages.append(nn.Sequential(*stages))

        # 평균 풀링과 Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dims[3] * block.expansion, num_classes)
        
    def forward_features(self, x):
        x = self.stem(x)
        for i in range(4):            
            x = self.stages[i](x)        
        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnet50():
    return ResNet(Bottleneck, dims=[64,128,256,512], depths=[3, 3, 9, 3], num_classes=100)
