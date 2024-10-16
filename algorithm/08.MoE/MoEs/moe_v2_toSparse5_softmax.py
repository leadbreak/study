import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.output_dim = output_dim
        
        # Gate Network
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 단일 nn.Linear 레이어로 모든 전문가의 출력을 통합
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
        # 전문가들 간의 대칭성을 깨기 위해 초기화
        nn.init.xavier_uniform_(self.experts.weight)
        nn.init.zeros_(self.experts.bias)
        
    def forward(self, x):
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        
        # Top-K 전문가 선택
        topk_probs, topk_indices = torch.topk(gate_logits, self.topk, dim=-1)  # [batch_size, topk]
        
        if self.topk > 1:
            # 선택된 Top-K 전문가의 logits을 재정규화 (softmax)
            topk_probs = F.softmax(topk_probs, dim=1)  # [batch_size, topk]
        else:
            # Top-K=1일 때는 가중치를 1로 설정
            topk_probs = torch.ones_like(topk_probs)  # [batch_size, 1]
        
        # 전문가 출력 계산
        expert_outputs = self.experts(x)  # [batch_size, num_experts * output_dim]
        # [batch_size, num_experts * output_dim] -> [batch_size, num_experts, output_dim]
        expert_outputs = expert_outputs.view(x.size(0), self.num_experts, self.output_dim)
        
        # Top-K 전문가의 출력 선택
        selected_expert_outputs = torch.gather(
            expert_outputs, 
            1, 
            topk_indices.unsqueeze(-1).expand(-1, -1, self.output_dim)
        )  # [batch_size, topk, output_dim]
        
        if self.topk > 1:
            # 가중 합산 using einsum
            output = torch.einsum('bk,bkd->bd', topk_probs, selected_expert_outputs)  # [batch_size, output_dim]
        else:
            # Top-K=1일 때는 선택된 전문가의 출력을 그대로 사용
            output = selected_expert_outputs.squeeze(1)  # [batch_size, output_dim]
        
        return output  # [batch_size, output_dim]

# 테스트를 위한 간단한 예제
if __name__ == "__main__":
    # 파라미터 설정
    input_dim = 10
    output_dim = 10
    num_experts = 5
    topk = 2  # topk=1로 변경하여 테스트 가능
    batch_size = 1
    
    # 모델 인스턴스화
    moe_model = MoE(input_dim, output_dim, num_experts, topk=topk)
    
    # 더미 입력 데이터
    x = torch.randn(batch_size, input_dim)
    
    # 모델 실행
    output = moe_model(x)
    print("모델 출력:", output)
    print("출력 형태:", output.shape)  # [batch_size, output_dim]
