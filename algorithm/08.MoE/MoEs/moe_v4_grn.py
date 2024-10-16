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

class MoE_GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,  num_experts=5, num_classes=10, topk=2, noise_std=0.1):
        super(MoE_GRN, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.output_dim = output_dim
        self.noise_std = noise_std  # 가우시안 노이즈의 표준편차

        # Gate Network
        self.gate = nn.Linear(input_dim, num_experts)

        # 전문가 레이어
        self.experts_fc1 = nn.Linear(input_dim, hidden_dim* num_experts)
        self.experts_fc2 = nn.Linear(hidden_dim* num_experts, output_dim * num_experts)
        
        # GRN 모듈
        self.grn = GRN(num_experts)
        
        # fc out 모듈
        self.classifier = nn.Linear(output_dim, num_classes)

        # 초기화
        nn.init.xavier_uniform_(self.experts_fc1.weight)
        nn.init.zeros_(self.experts_fc1.bias)
        
        nn.init.xavier_uniform_(self.experts_fc2.weight)
        nn.init.zeros_(self.experts_fc2.bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # 입력 이미지 평탄화
        x = x.view(x.size(0), -1)  # [batch_size, 3072]
        
        # Gate logits 계산
        gate_logits = self.gate(x)  # [batch_size, num_experts]

        # 가우시안 노이즈 추가
        if self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise  # 노이즈 추가

        # GRN 적용
        gate_logits = self.grn(gate_logits)

        # Gate 확률 계산 (softmax)
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # Top-K 전문가 선택
        topk_probs, topk_indices = torch.topk(gate_probs, self.topk, dim=-1)  # [batch_size, topk]

        # 전문가 출력 계산
        expert_outputs = self.experts_fc1(x)  # [batch_size, num_experts * hidden_dim]
        expert_outputs = F.relu(expert_outputs)  # [batch_size, num_experts * hidden_dim]
        expert_outputs = self.experts_fc2(expert_outputs)  # [batch_size, num_experts * output_dim]        
        expert_outputs = expert_outputs.view(-1, self.num_experts, self.output_dim)  # [batch_size, num_experts, output_dim]

        # 선택된 전문가의 출력 추출
        selected_expert_outputs = torch.gather(
            expert_outputs,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.output_dim)
        )  # [batch_size, topk, output_dim]

        # 최종 출력 계산
        output = torch.einsum('bk,bkd->bd', topk_probs, selected_expert_outputs)  # [batch_size, output_dim]

        return self.classifier(output)

# 테스트 코드
if __name__ == "__main__":
    input_dim = 10
    output_dim = 10
    num_experts = 5
    topk = 2
    batch_size = 2  
    noise_std = 0.1

    moe_model = MoE_GRN(input_dim, output_dim, num_experts, topk=topk, noise_std=noise_std)

    x = torch.randn(batch_size, input_dim)
    output = moe_model(x)
    print("출력:", output)
    print("출력 크기:", output.shape)