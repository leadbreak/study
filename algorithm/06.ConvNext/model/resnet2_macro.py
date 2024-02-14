import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4  # 확장 비율, Bottleneck 구조에서 마지막 Conv Layer의 출력 채널이 입력 채널의 4배가 됨

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 첫 번째 1x1 Convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 두 번째 3x3 Convolution, stride는 주어진 값 사용
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 세 번째 1x1 Convolution, 채널 확장
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 입력의 형상을 출력의 형상에 맞춰주기 위한 downsampling layer

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, 
                 block, 
                 dims=[64,128,256,512],
                 layers=[3,3,9,3],
                 num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # downsampling layers
        self.downsample_layers = nn.ModuleList()
        patchify_stem = nn.Sequential(
                            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                            nn.BatchNorm2d(dims[0])
                        )
        self.downsample_layers.append(patchify_stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                                nn.BatchNorm2d(dims[i+1])
                                )
            self.downsample_layers.append(downsample_layer)
            
        # stage layer
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                        *[block(dims[i]) for _ in range(layers[i])]
                    )
            
            self.stages.append(stage)       
        
        
        
        # 초기 Convolution과 Max Pooling Layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Residual Blocks
        self.layer1 = self._make_layer(block, dims[0], layers[0])
        self.layer2 = self._make_layer(block, dims[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, dims[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, dims[3], layers[3], stride=2)
        # 평균 풀링과 Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels*8 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 각 Residual Block을 통과한 후의 형상 출력
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=100)

def resnet50_stages():
    return ResNet(Bottleneck, [3, 3, 9, 3], num_classes=100)
