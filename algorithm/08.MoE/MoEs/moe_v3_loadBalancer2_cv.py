import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2, noise_std=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.output_dim = output_dim
        self.noise_std = noise_std  # 가우시안 노이즈의 표준편차

        # Gate Network
        self.gate = nn.Linear(input_dim, num_experts)

        # 전문가 레이어
        self.experts = nn.Linear(input_dim, output_dim * num_experts)

        # 초기화
        nn.init.xavier_uniform_(self.experts.weight)
        nn.init.zeros_(self.experts.bias)

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

        return output, load_balance_loss

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
    output, load_balance_loss = moe_model(x)
    print("출력:", output)
    print("출력 크기:", output.shape)
    print("로드 밸런싱 손실:", load_balance_loss.item())
