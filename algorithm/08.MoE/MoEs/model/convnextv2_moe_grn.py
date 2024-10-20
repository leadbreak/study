import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, dim]
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)  # [batch_size, 1]
        Nx = Gx / (Gx.mean(dim=0, keepdim=True) + self.eps)  # [batch_size, 1]
        x = self.gamma * (x * Nx) + self.beta
        return x

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

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2, noise_std=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.output_dim = output_dim
        self.noise_std = noise_std

        # Gate Network
        self.gate = nn.Linear(input_dim, num_experts)

        # 전문가 레이어
        self.experts = nn.Linear(input_dim, output_dim * num_experts)

        # GRN 모듈
        self.grn = GRN(num_experts)

        # 초기화
        nn.init.xavier_uniform_(self.experts.weight)
        nn.init.zeros_(self.experts.bias)

    def forward(self, x):
        # Gate logits 계산
        gate_logits = self.gate(x)  # [batch_size, num_experts]

        # 가우시안 노이즈 추가
        if self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # GRN 적용
        gate_logits = self.grn(gate_logits)

        # Gate 확률 계산 (softmax)
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # Top-K 전문가 선택
        topk_probs, topk_indices = torch.topk(gate_probs, self.topk, dim=-1)  # [batch_size, topk]

        # 전문가 출력 계산
        expert_outputs = self.experts(x)  # [batch_size, num_experts * output_dim]
        expert_outputs = expert_outputs.view(-1, self.num_experts, self.output_dim)  # [batch_size, num_experts, output_dim]

        # 선택된 전문가의 출력 추출
        selected_expert_outputs = torch.gather(
            expert_outputs,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.output_dim)
        )  # [batch_size, topk, output_dim]

        # 최종 출력 계산
        output = torch.einsum('bk,bkd->bd', topk_probs, selected_expert_outputs)  # [batch_size, output_dim]

        return output


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


class ConvNeXtV2_MoE_GRN(nn.Module):
    def __init__(self,  
                 in_chans=3,
                 dims=[96,192,384,768],
                 depths=[3,3,9,3], 
                 droppath=0.1,
                 num_classes=100):
        super(ConvNeXtV2_MoE_GRN, self).__init__()
        
        # Patchify Stem
        stem = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)),
            ('stem_ln', LayerNorm(dims[0])),
        ]))
        
        # Downsample layers
        self.downsample_layers = nn.ModuleList()    
        self.downsample_layers.append(stem)    
        
        for i in range(3):
            downsample_layer = nn.Sequential(OrderedDict([
                                (f'ds_ln', LayerNorm(dims[i])),
                                (f'ds_conv', nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)),                                
                                ]))
            self.downsample_layers.append(downsample_layer)
        
        # Stage layers
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, droppath, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.ModuleList(
                [Block(dims[i], 
                        dp_rate=dp_rates[cur+j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            
            cur += depths[i]

        # Average pooling and Fully Connected Layer (MoE로 대체)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layernorm = nn.LayerNorm(dims[3])      # Channel Last Layernorm
        
        # 최종 분류기만 MoE로 유지
        self.moe = MoE(input_dim=dims[3], output_dim=dims[3], num_experts=8, topk=1, noise_std=0.1)
        self.classifier = nn.Linear(dims[3], num_classes)
        
    def forward_features(self, x):
        for i in range(4):        
            x = self.downsample_layers[i](x)    
            for blk in self.stages[i]:
                x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.avgpool(x)     # [N, C, 1, 1]
        x = torch.flatten(x, 1) # [N, C]
        x = self.layernorm(x)
        
        x = self.moe(x)  # [N, N]
        
        return self.classifier(x) # [N, num_classes]

def load_convNext_moe_grn(dims=[96,192,384,768], depths=[3, 3, 9, 3], num_classes=100, **args):
    return ConvNeXtV2_MoE_GRN(dims=dims, depths=depths, num_classes=num_classes, **args)

# Test code
if __name__ == "__main__":
    model = load_convNext_moe_grn()
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print("Logits shape:", logits.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
