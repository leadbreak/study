'''
기본적인 Sparse MoE 구현2
ModuleList에 의한 전문가 관리로 비효율적임
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        gate_logits = self.layer(x)  # [batch_size, num_experts]
        gate_output = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
        return gate_output

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        gate_output = self.gate(x)  # [batch_size, num_experts]
        
        # 가장 큰 게이트 확률을 가진 상위 k개의 전문가 선택
        topk_probs, topk_indices = torch.topk(gate_output, self.topk, dim=-1)  # [batch_size, topk]
        
        # 출력값을 초기화 - 선택되지 않은 전문가 영역들은 0으로 출력
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.experts[0].layer.out_features, device=x.device)
        
        # 선택된 전문가들의 의견만 덮어쓰기
        for i in range(self.topk):
            expert_idx = topk_indices[:, i]  # [batch_size]
            expert_prob = topk_probs[:, i].unsqueeze(1)  # [batch_size, 1]
            
            # 각 배치에 대해 전문가의 출력을 계산하고 가중합
            expert_outputs = torch.stack([self.experts[idx](x[b]) for b, idx in enumerate(expert_idx)], dim=0)  # [batch_size, output_dim]
            output += expert_outputs * expert_prob  # [batch_size, output_dim]
        
        return output  # [batch_size, output_dim]

# 테스트를 위한 간단한 예제
if __name__ == "__main__":
    # 파라미터 설정
    input_dim = 10
    output_dim = 10
    num_experts = 5
    topk = 2
    batch_size = 1
    
    # 모델 인스턴스화
    moe_model = MoE(input_dim, output_dim, num_experts, topk=topk)
    
    # 더미 입력 데이터
    x = torch.randn(batch_size, input_dim)
    
    # 모델 실행
    output = moe_model(x)
    print("모델 출력:", output)
    print("출력 형태:", output.shape)  # [batch_size, output_dim]
