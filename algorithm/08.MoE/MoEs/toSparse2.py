import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        return torch.softmax(self.layer(x), dim=-1)  # [batch size, num experts]

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        
        # 전문가의 가중치와 편향을 하나의 파라미터로 정의
        # 전문가 수 x (입력 차원 x 출력 차원)
        self.experts_weights = nn.Parameter(torch.randn(num_experts, input_dim, output_dim))
        self.experts_bias = nn.Parameter(torch.zeros(num_experts, output_dim))
        
        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        batch_size = x.size(0)
        
        # 게이트웨이 네트워크를 통해 각 전문가의 선택 확률 계산
        gate_outputs = self.gate(x)  # [batch_size, num_experts]
        
        # 상위 topk개의 전문가 선택
        top_k_values, top_k_indices = torch.topk(gate_outputs, self.topk, dim=-1)  # [batch_size, topk]
        
        # 모든 전문가의 출력을 한 번에 계산
        # x: [batch_size, input_dim]
        # experts_weights: [num_experts, input_dim, output_dim]
        # expert_outputs: [batch_size, num_experts, output_dim]
        expert_outputs = torch.einsum('bi,eio->bei', x, self.experts_weights) + self.experts_bias  # [batch_size, num_experts, output_dim]
        
        # 선택된 topk 전문가의 출력을 추출
        # top_k_indices: [batch_size, topk]
        # top_k_indices_expanded: [batch_size, topk, output_dim]
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))  # [batch_size, topk, output_dim]
        selected_expert_outputs = torch.gather(expert_outputs, 1, top_k_indices_expanded)  # [batch_size, topk, output_dim]
        
        # 선택된 전문가의 가중치를 곱함
        top_k_values = top_k_values.unsqueeze(-1)  # [batch_size, topk, 1]
        weighted_expert_outputs = selected_expert_outputs * top_k_values  # [batch_size, topk, output_dim]
        
        # 최종 출력은 topk 전문가의 가중합
        output = weighted_expert_outputs.sum(dim=1)  # [batch_size, output_dim]
        
        return output