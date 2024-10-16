'''
기본적인 Sparse MoE 구현
topk1 only & ModuleList에 의한 전문가 관리로 비효율적임
'''

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
        
        # 가장 큰 게이트 확률을 가진 상위 1개의 전문가 선택
        top_k_values, top_k_indices = torch.topk(gate_outputs, 1, dim=-1) # [batch size, 1]
        selected_expert_idx = top_k_indices.squeeze(-1) # [batch size]
        # print(f'Selected Expert ID: {selected_expert_idx}')
        
        # 선택된 전문가의 출력을 가져오기
        batch_size = x.size(0)
        output = torch.stack([self.experts[selected_expert_idx[i]](x[i]) for i in range(batch_size)], dim=0) # [batch size, output dim]
        
        return output