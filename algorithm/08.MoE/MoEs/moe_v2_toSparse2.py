import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        gate_logits = self.layer(x)  # [batch_size, num_experts]
        gate_probs = torch.softmax(gate_logits, dim=-1)  # SoftMax 적용하여 확률 분포 생성
        return gate_probs  # [batch_size, num_experts]

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk

        # 모든 전문가를 하나의 큰 선형 계층으로 통합
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        nn.init.xavier_uniform_(self.experts.weight)
        nn.init.zeros_(self.experts.bias)

        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        batch_size = x.size(0)
        output_dim = self.experts.out_features // self.num_experts  # 각 전문가의 출력 차원

        # 1. 게이팅 네트워크를 통해 각 전문가의 선택 확률 계산
        gate_probs = self.gate(x)  # [batch_size, num_experts]

        # 2. 상위 topk개의 전문가 선택
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.topk, dim=-1)  # [batch_size, topk]

        # 3. 모든 전문가의 출력을 한 번에 계산
        # 전문가들의 출력은 [batch_size, num_experts * output_dim]에서 [batch_size, num_experts, output_dim]으로 재배열
        expert_outputs = self.experts(x)  # [batch_size, num_experts * output_dim]
        expert_outputs = expert_outputs.view(batch_size, self.num_experts, output_dim)  # [batch_size, num_experts, output_dim]

        # 4. 선택된 topk 전문가의 출력을 추출
        # top_k_indices는 [batch_size, topk], 이를 이용해 배치별로 topk 전문가의 출력을 gather
        # 먼저, 배치 인덱스를 생성하여 advanced indexing을 수행
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.topk).to(x.device)  # [batch_size, topk]
        selected_expert_outputs = expert_outputs[batch_indices, top_k_indices]  # [batch_size, topk, output_dim]

        # 5. 선택된 전문가의 가중치를 곱함
        top_k_probs = top_k_probs.unsqueeze(-1)  # [batch_size, topk, 1]
        weighted_expert_outputs = selected_expert_outputs * top_k_probs  # [batch_size, topk, output_dim]

        # 6. 최종 출력은 topk 전문가의 가중합
        output = weighted_expert_outputs.sum(dim=1)  # [batch_size, output_dim]

        return output  # [batch_size, output_dim]

# 예시 사용법
if __name__ == "__main__":
    input_dim = 128
    output_dim = 256
    num_experts = 10
    batch_size = 32
    topk = 2

    model = MoE(input_dim, output_dim, num_experts, topk=topk)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(output.shape)  # 예상 출력: [32, 256]
