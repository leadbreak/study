import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim, ndim=4, eps=1e-6):
        super(GRN, self).__init__()
        self.ndim = ndim
        if ndim == 4:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        elif ndim == 2:
            self.gamma = nn.Parameter(torch.ones(1, dim))
            self.beta = nn.Parameter(torch.zeros(1, dim))
        else:
            raise ValueError("ndim must be 2 or 4")
        self.eps = eps

    def forward(self, x):
        if self.ndim == 4:
            # x: [N, H, W, C]
            Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)  # [N, 1, 1, C]
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)  # [N, 1, 1, C]
            x = self.gamma * (x * Nx) + self.beta + x
        elif self.ndim == 2:
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
        self.noise_std = noise_std
        self.output_dim = output_dim // num_experts  # Adjusted to divide by num_experts

        assert output_dim % num_experts == 0, f"output_dim({output_dim}) should be divisible by num_experts({num_experts})"

        # Experts
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, self.output_dim)
            for _ in range(num_experts)
        ])

        # Gate Network
        self.gate = nn.Linear(input_dim, num_experts)

        # GRN module (for 2D input)
        self.grn = GRN(num_experts, ndim=2)

    def forward(self, x):
        # x: [batch_size, input_dim]
        # Gate logits
        gate_logits = self.gate(x)  # [batch_size, num_experts]

        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # Apply GRN
        gate_logits = self.grn(gate_logits)  # [batch_size, num_experts]

        # Gate probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, output_dim_per_expert]

        # Gate probabilities applied to expert outputs
        gate_probs = gate_probs.unsqueeze(-1)  # [batch_size, num_experts, 1]
        x = (expert_outputs * gate_probs).sum(dim=1)  # [batch_size, output_dim_per_expert]

        # Restore output dimension
        x = x.repeat(1, self.num_experts)  # [batch_size, output_dim]
        return x

class Block(nn.Module):

    def __init__(self, dim, dp_rate):
        super(Block, self).__init__()
        
        self.dim = dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = nn.LayerNorm(dim)

        # Adjusted output_dim to match parameter counts
        self.moe = MoE(input_dim=dim, output_dim=dim * 4, num_experts=2, topk=1, noise_std=0.1)

        self.act = QuickGELU()
        self.grn = GRN(dim * 4, ndim=4)  # Adjusted GRN to match output dimension
        self.pwconv2 = nn.Linear(dim * 4, dim)
        
        # droppath(stochastic depth)
        self.droppath = DropPath(dp_rate) if dp_rate > 0. else nn.Identity()

    def forward(self, x):
        identity = x  # [N, C, H, W]
        x = self.dwconv(x)  # [N, C, H, W]
        x = x.permute(0,2,3,1)  # [N, H, W, C]
        x = self.layernorm(x)  # [N, H, W, C]

        N, H, W, C = x.shape
        x = x.reshape(-1, C)  # [N * H * W, C]
        x = self.moe(x)  # [N * H * W, dim * 4]
        x = x.reshape(N, H, W, -1)  # [N, H, W, dim * 4]

        x = self.act(x)
        x = self.grn(x)  # [N, H, W, dim * 4]
        x = self.pwconv2(x)  # [N, H, W, dim]
        x = x.permute(0,3,1,2)  # [N, C, H, W]

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

        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layernorm = nn.LayerNorm(dims[3])      # Channel Last Layernorm

        # Adjusted MoE classifier
        self.moe_fc = MoE(input_dim=dims[3], output_dim=num_classes, num_experts=1, topk=1, noise_std=0.1)
        
    def forward_features(self, x):
        for i in range(4):        
            x = self.downsample_layers[i](x)    
            for blk in self.stages[i]:
                x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.avgpool(x)     # (N, C, 1, 1)
        x = torch.flatten(x, 1) # (N, C)
        x = self.layernorm(x)
        
        x = self.moe_fc(x)
        
        return x
        
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
