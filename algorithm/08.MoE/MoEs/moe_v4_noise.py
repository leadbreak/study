import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, add_noise=False, noise_std=1.0):
        super(GatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        self.add_noise = add_noise
        self.noise_std = noise_std

    def forward(self, x):
        gate_logits = self.layer(x)  # [batch_size, num_experts]
        if self.add_noise and self.training:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        return gate_logits  # 로짓을 반환 (SoftMax는 MoE 클래스에서 수행)

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()

        self.layer = nn.Linear(input_dim, output_dim)
        # 필요한 경우 활성화 함수를 추가 가능
        # self.activation = nn.ReLU()
            
    def forward(self, x):
        # return self.activation(self.layer(x))
        return self.layer(x)

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, topk=2, add_noise=False, noise_std=1.0, lambda_aux=1.0):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.lambda_aux = lambda_aux  # 보조 손실의 가중치

        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts, add_noise=add_noise, noise_std=noise_std)
            
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        batch_size = x.size(0)
        output_dim = self.experts[0].layer.out_features
                
        # 게이트웨이 네트워크를 통해 각 전문가의 로짓 계산
        gate_logits = self.gate(x)  # [batch_size, num_experts]

        # 상위 topk개의 전문가 선택
        top_k_values, top_k_indices = torch.topk(gate_logits, self.topk, dim=-1)  # [batch_size, topk]

        # 비-topk 전문가의 로짓을 -∞으로 설정
        mask = torch.ones_like(gate_logits, dtype=torch.bool)  # [batch_size, num_experts]
        mask.scatter_(1, top_k_indices, False)  # 상위 topk 위치를 False로 설정
        gate_logits.masked_fill_(mask, float('-inf'))  # 비-topk 위치를 -∞으로 설정

        # SoftMax 적용하여 확률 분포 생성
        gate_probs = torch.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # 상위 topk개의 전문가의 확률만 남아 있음 (나머지는 0)
        top_k_probs = torch.gather(gate_probs, 1, top_k_indices)  # [batch_size, topk]

        # 선택된 전문가들의 출력을 저장할 텐서를 초기화
        selected_expert_outputs = torch.zeros(batch_size, self.topk, output_dim, device=x.device)
        
        # 각 topk에 대해 전문가의 출력을 계산
        for i in range(self.topk):
            # 현재 topk 위치에 해당하는 전문가 인덱스
            expert_indices = top_k_indices[:, i]  # [batch_size]
            expert_probs = top_k_probs[:, i]      # [batch_size]
            
            # 전문가별로 입력을 모아서 계산
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)  # [batch_size]
                if not mask.any():
                    continue
                indices = mask.nonzero(as_tuple=False).squeeze()  # 선택된 배치 인덱스
                x_selected = x[indices]  # [selected_batch_size, input_dim]
                expert = self.experts[expert_idx]
                expert_output = expert(x_selected)  # [selected_batch_size, output_dim]
                selected_expert_outputs[indices, i, :] = expert_output

        # 선택된 전문가의 가중치를 곱함
        top_k_probs = top_k_probs.unsqueeze(-1)  # [batch_size, topk, 1]
        weighted_expert_outputs = selected_expert_outputs * top_k_probs  # [batch_size, topk, output_dim]

        # 최종 출력은 topk 전문가의 가중합
        output = weighted_expert_outputs.sum(dim=1)  # [batch_size, output_dim]

        # 부하 균형을 위한 보조 손실 계산
        # 각 전문가에 대한 선택 확률의 합계 계산
        expert_load = gate_probs.sum(dim=0)  # [num_experts]

        # 평균과 표준편차 계산
        load_mean = expert_load.mean()
        load_std = expert_load.std()

        # 변동 계수 (Coefficient of Variation) 계산
        cv = load_std / load_mean

        # 보조 손실은 변동 계수
        aux_loss = cv

        # 전체 손실은 메인 손실 + lambda_aux * aux_loss
        # 여기서는 MoE가 보조 손실을 반환하므로, 훈련 루프에서 이를 사용하여 전체 손실을 계산해야 함
                
        return output, self.lambda_aux * aux_loss

# 예시 사용법
if __name__ == "__main__":
    input_dim = 128
    output_dim = 256
    num_experts = 10
    batch_size = 32
    topk = 2
    add_noise = True
    noise_std = 1.0
    lambda_aux = 0.1  # 보조 손실의 가중치

    model = MoE(input_dim, output_dim, num_experts, topk=topk, add_noise=add_noise, noise_std=noise_std, lambda_aux=lambda_aux)
    x = torch.randn(batch_size, input_dim)
    output, aux_loss = model(x)
    print(output.shape)  # 예상 출력: [32, 256]
    print(f"Auxiliary Loss: {aux_loss.item()}")  # 보조 손실 출력
