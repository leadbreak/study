import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        return torch.softmax(self.layer(x), dim=-1)  # [batch size, num experts]

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        
        # 전문가들의 가중치와 편향을 하나의 파라미터 텐서로 정의
        # [num_experts, input_dim, output_dim]
        self.experts_weights = nn.Parameter(torch.randn(num_experts, input_dim, output_dim))
        # [num_experts, output_dim]
        self.experts_bias = nn.Parameter(torch.zeros(num_experts, output_dim))
        
        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        batch_size = x.size(0)
        
        # 게이트웨이 네트워크를 통해 각 전문가의 선택 확률 계산
        gate_outputs = self.gate(x)  # [batch_size, num_experts]
        
        # 모든 전문가의 출력을 한 번에 계산
        # expert_outputs: [batch_size, num_experts, output_dim]
        expert_outputs = torch.einsum('bi,eio->bei', x, self.experts_weights) + self.experts_bias  # [batch_size, num_experts, output_dim]
        
        # 게이트웨이 확률과 전문가 출력을 가중합산
        gate_outputs = gate_outputs.unsqueeze(-1)  # [batch_size, num_experts, 1]
        output = torch.sum(gate_outputs * expert_outputs, dim=1)  # [batch_size, output_dim]
        
        return output

# 예시 사용법
if __name__ == "__main__":
    input_dim = 128
    output_dim = 256
    num_experts = 10
    batch_size = 32

    model = MoE(input_dim, output_dim, num_experts)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(output.shape)  # 예상 출력: [32, 256]
