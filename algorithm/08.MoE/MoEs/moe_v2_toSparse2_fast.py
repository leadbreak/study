import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        return torch.softmax(self.layer(x), dim=-1)  # [batch_size, num_experts]

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        
        # 전문가의 가중치와 편향을 정의
        self.experts_weights = nn.Parameter(torch.randn(num_experts, input_dim, output_dim))
        self.experts_bias = nn.Parameter(torch.zeros(num_experts, output_dim))
        
        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        batch_size = x.size(0)
        input_dim = x.size(1)
        output_dim = self.experts_bias.size(1)
        
        # 게이트웨이 네트워크를 통해 각 전문가의 선택 확률 계산
        gate_outputs = self.gate(x)  # [batch_size, num_experts]
        
        # 상위 topk개의 전문가 선택
        top_k_values, top_k_indices = torch.topk(gate_outputs, self.topk, dim=-1)  # [batch_size, topk]
        
        # 선택된 전문가의 가중치와 편향 추출
        # top_k_indices: [batch_size, topk]
        # experts_weights: [num_experts, input_dim, output_dim]
        # selected_experts_weights: [batch_size, topk, input_dim, output_dim]
        selected_experts_weights = self.experts_weights[top_k_indices]  # 인덱싱을 통해 선택
        selected_experts_bias = self.experts_bias[top_k_indices]        # [batch_size, topk, output_dim]
        
        # 입력 x를 확장하여 선택된 전문가들과 매트릭스 곱셈이 가능하도록 함
        # x_expanded: [batch_size, topk, input_dim]
        x_expanded = x.unsqueeze(1).expand(-1, self.topk, -1)
        
        # 전문가 출력을 계산
        # [batch_size, topk, output_dim]
        selected_expert_outputs = torch.einsum('bki,bkio->bko', x_expanded, selected_experts_weights) + selected_experts_bias
        
        # 선택된 전문가의 가중치를 곱함
        top_k_values = top_k_values.unsqueeze(-1)  # [batch_size, topk, 1]
        weighted_expert_outputs = selected_expert_outputs * top_k_values  # [batch_size, topk, output_dim]
        
        # 최종 출력은 topk 전문가의 가중합
        output = weighted_expert_outputs.sum(dim=1)  # [batch_size, output_dim]
        
        return output

# 예시 사용법
if __name__ == "__main__":
    input_dim = 128
    output_dim = 256
    num_experts = 100  # 전문가 수를 크게 설정하여 효율성 확인
    batch_size = 32
    topk = 4  # 선택할 전문가 수

    model = MoE(input_dim, output_dim, num_experts, topk=topk)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(output.shape)  # 예상 출력: [32, 256]
