import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.layer(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        return torch.softmax(self.layer(x), dim=-1) # [batch size, num experts]

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim) for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        gate_outputs = self.gate(x) # [batch size, num experts]
        
        # 전문가들의 출력을 병렬로 계산
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # [batch size, num experts, output dim]
        
        # 게이트웨이 확률과 전문가 출력을 가중합산
        gate_outputs = gate_outputs.unsqueeze(-1) # [batch size, num experts, 1]
        output = torch.sum(gate_outputs * expert_outputs, dim=1) # [batch size, output dim]
        
        return output