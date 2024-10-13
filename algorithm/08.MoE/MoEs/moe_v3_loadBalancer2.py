import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        # GRN을 추가합니다.
        self.grn = GRN(num_experts)
        
    def forward(self, x):
        gate_logits = self.layer(x)  # [batch_size, num_experts]
        # GRN을 적용하여 게이트 출력 정규화
        gate_logits = self.grn(gate_logits.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        gate_outputs = torch.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
        return gate_outputs

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        # 필요한 경우 활성화 함수를 추가할 수 있습니다.
        # self.activation = nn.ReLU()
        
    def forward(self, x):
        # return self.activation(self.layer(x))
        return self.layer(x)

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        
        # 전문가들을 nn.ModuleList로 관리합니다.
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        batch_size = x.size(0)
        output_dim = self.experts[0].layer.out_features
        
        # 게이트웨이 네트워크를 통해 각 전문가의 선택 확률 계산
        gate_outputs = self.gate(x)  # [batch_size, num_experts]
        
        # 상위 topk개의 전문가 선택
        top_k_values, top_k_indices = torch.topk(gate_outputs, self.topk, dim=-1)  # [batch_size, topk]
        
        # 선택된 전문가들의 출력을 저장할 텐서를 초기화합니다.
        selected_expert_outputs = torch.zeros(batch_size, self.topk, output_dim, device=x.device)
        
        # 각 전문가에 대해 입력을 그룹화하여 계산
        for i in range(self.topk):
            # 현재 topk 위치에 해당하는 전문가 인덱스
            expert_indices = top_k_indices[:, i]  # [batch_size]
            expert_weights = top_k_values[:, i]   # [batch_size]
            
            # 전문가별로 입력을 모아서 계산합니다.
            unique_expert_indices = expert_indices.unique()
            for expert_idx in unique_expert_indices:
                mask = (expert_indices == expert_idx)  # [batch_size]
                indices = mask.nonzero(as_tuple=False).squeeze()  # 선택된 배치 인덱스
                
                if indices.numel() == 0:
                    continue
                
                # 선택된 입력들
                x_selected = x[indices]  # [selected_batch_size, input_dim]
                # 해당 전문가의 출력 계산
                expert = self.experts[expert_idx]
                expert_output = expert(x_selected)  # [selected_batch_size, output_dim]
                
                # 출력 저장
                selected_expert_outputs[indices, i, :] = expert_output

        # 선택된 전문가의 가중치를 곱함
        top_k_values = top_k_values.unsqueeze(-1)  # [batch_size, topk, 1]
        weighted_expert_outputs = selected_expert_outputs * top_k_values  # [batch_size, topk, output_dim]
        
        # 최종 출력은 topk 전문가의 가중합
        output = weighted_expert_outputs.sum(dim=1)  # [batch_size, output_dim]
        
        return output

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        # x: [batch_size, num_experts]
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)  # [batch_size, 1]
        Nx = Gx / (Gx.mean(dim=0, keepdim=True) + 1e-6)  # [batch_size, 1]
        return self.gamma * (x * Nx) + self.beta + x

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
