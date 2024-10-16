import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 테스트 코드
if __name__ == "__main__":
    input_dim = 10
    output_dim = 10
    num_experts = 5
    topk = 2
    batch_size = 2  
    noise_std = 0.1

    moe_model = MoE(input_dim, output_dim, num_experts, topk=topk, noise_std=noise_std)

    x = torch.randn(batch_size, input_dim)
    output = moe_model(x)
    print("출력:", output)
    print("출력 크기:", output.shape)
