import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x: [N, H, W, C]
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)  # [N, 1, 1, C]
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)  # [N, 1, 1, C]
        x = self.gamma * (x * Nx) + self.beta + x
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
        # x: [N, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True, unbiased=False).sqrt()
        x = (x - mean) / (std + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=2, topk=1, noise_std=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.output_dim = output_dim // num_experts
        self.noise_std = noise_std  # 가우시안 노이즈의 표준편차

        assert output_dim % num_experts == 0, f"output_dim({output_dim}) should be divisible by num_experts({num_experts})"

        # Gate Network
        self.gate = nn.Linear(input_dim, num_experts)

        # 전문가 레이어
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, self.output_dim)
            for _ in range(num_experts)
        ])

        # 초기화
        for expert in self.experts:
            nn.init.xavier_uniform_(expert.weight)
            nn.init.zeros_(expert.bias)

    def forward(self, x):
        # Gate logits 계산
        gate_logits = self.gate(x)  # [batch_size, num_experts]

        # 가우시안 노이즈 추가
        if self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise  # 노이즈 추가

        # Gate 확률 계산 (softmax)
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # -----------------------------
        # 로드 밸런싱 손실 계산
        # -----------------------------

        # 배치에 대해 전문가 확률 평균 계산
        gate_probs_mean = gate_probs.mean(dim=0)  # [num_experts]

        # 균일 분포 기대값
        expected_prob = 1.0 / self.num_experts

        # 로드 밸런싱 손실 계산
        load_balance_loss = ((gate_probs_mean - expected_prob) ** 2).mean() * self.num_experts

        # -----------------------------
        # Top-K 전문가 선택 및 출력 계산
        # -----------------------------

        # Top-K 전문가 선택
        topk_probs, topk_indices = torch.topk(gate_probs, self.topk, dim=-1)  # [batch_size, topk]

        # 각 전문가의 출력 계산
        expert_outputs = []
        for idx in range(self.num_experts):
            expert_outputs.append(self.experts[idx](x))  # [batch_size, output_dim_per_expert]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim_per_expert]

        # 선택된 전문가의 출력 추출
        selected_expert_outputs = torch.gather(
            expert_outputs,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.output_dim)
        )  # [batch_size, topk, output_dim_per_expert]

        # 최종 출력 계산
        output = torch.einsum('bk,bkd->bd', topk_probs, selected_expert_outputs)  # [batch_size, output_dim_per_expert]

        # 출력 차원을 복원
        output = output.repeat(1, self.num_experts)  # [batch_size, output_dim]

        return output, load_balance_loss

class Block(nn.Module):

    def __init__(self, dim, dp_rate):
        super(Block, self).__init__()
        
        self.dim = dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = nn.LayerNorm(dim)
        self.moe = MoE(input_dim=dim, output_dim=dim*4, num_experts=2, topk=1, noise_std=0.1)
        self.act = QuickGELU()
        self.grn = GRN(dim*4)  # Global Response Normalization
        self.pwconv2 = nn.Linear(dim*4, dim)
        
        # droppath(stochastic depth)
        self.droppath = DropPath(dp_rate) if dp_rate > 0. else nn.Identity()

    def forward(self, x):
        identity = x  # [N, C, H, W]
        x = self.dwconv(x)  # [N, C, H, W]
        x = x.permute(0,2,3,1)  # [N, H, W, C]
        x = self.layernorm(x)  # [N, H, W, C]

        N, H, W, C = x.shape
        x = x.reshape(-1, C)  # [N * H * W, C]
        x, lb_loss = self.moe(x)  # [N * H * W, dim * 4]
        x = x.reshape(N, H, W, -1)  # [N, H, W, dim * 4]

        x = self.act(x)
        x = self.grn(x)  # [N, H, W, dim * 4]
        x = self.pwconv2(x)  # [N, H, W, dim]
        x = x.permute(0,3,1,2)  # [N, C, H, W]

        x = identity + self.droppath(x)

        return x, lb_loss

class ConvNeXtV2_MoE(nn.Module):
    def __init__(self,  
                 in_chans=3,
                 dims=[96,192,384,768],
                 depths=[3,3,9,3], 
                 droppath=0.1,
                 num_classes=100):
        super(ConvNeXtV2_MoE, self).__init__()
        
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
            stage = nn.ModuleList(
                [Block(dims[i], 
                        dp_rate=dp_rates[cur+j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            
            cur += depths[i]

        # 평균 풀링과 Fully Connected Layer (MoE로 대체)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layernorm = nn.LayerNorm(dims[3])      # Channel Last Layernorm
        
        # MoE로 최종 분류기 대체
        self.moe_fc = MoE(input_dim=dims[3], output_dim=num_classes, num_experts=1, topk=1, noise_std=0.1)
        
    def forward_features(self, x):
        lb_losses = []
        for i in range(4):        
            x = self.downsample_layers[i](x)    
            for blk in self.stages[i]:
                x, lb_loss = blk(x)
                lb_losses.append(lb_loss)
        return x, lb_losses

    def forward(self, x):
        x, lb_losses = self.forward_features(x)

        x = self.avgpool(x)     # (N, C, 1, 1)
        x = torch.flatten(x, 1) # (N, C)
        x = self.layernorm(x)
        
        x, fc_lb_loss = self.moe_fc(x)
        lb_losses.append(fc_lb_loss)
        
        total_lb_loss = torch.stack(lb_losses).mean()
        return x, total_lb_loss
    
def load_convNext_moe(dims=[96,192,384,768], depths=[3, 3, 9, 3], num_classes=100, **args):
    return ConvNeXtV2_MoE(dims=dims, depths=depths, num_classes=num_classes, **args)

# 테스트 코드
if __name__ == "__main__":
    model = load_convNext_moe()
    x = torch.randn(2, 3, 224, 224)
    logits, lb_loss = model(x)
    print("Logits shape:", logits.shape)
    print("Load balancing loss:", lb_loss.item())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
