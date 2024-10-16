import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.output_dim = output_dim
        
        # 단일 nn.Linear 레이어로 모든 전문가의 출력을 통합
        self.experts = nn.Linear(input_dim, output_dim * num_experts)
        
        # 전문가들 간의 대칭성을 깨기(Breaking Symmetry) 위해 초기화
        nn.init.xavier_uniform_(self.experts.weight)
        nn.init.zeros_(self.experts.bias)
        
        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        gate_output = self.gate(x)  # [batch_size, num_experts]
        
        # Top-K 전문가 선택
        topk_probs, topk_indices = torch.topk(gate_output, self.topk, dim=-1)  # [batch_size, topk]
        
        # 전문가 출력 계산
        expert_outputs = self.experts(x)  # [batch_size, num_experts * output_dim]
         # view : [batch_size, num_experts * output_dim] -> [batch_size, num_experts, output_dim]
        expert_outputs = expert_outputs.view(-1, self.num_experts, self.output_dim) 
        
        # Top-K 전문가의 출력 선택
        selected_expert_outputs = torch.gather(
            expert_outputs, 
            1, 
            topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
        )  # [batch_size, topk, output_dim]
        
        # 가중 합산
        output = torch.einsum('bk,bkd->bd', topk_probs, selected_expert_outputs)  # [batch_size, output_dim]
                
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
