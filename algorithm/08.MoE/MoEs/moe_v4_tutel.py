''' tutel parameters docs
* Usage of MOELayer Args:

        gate_type        : dict-type gate description, e.g. {'type': 'top', 'k': 2, 'capacity_factor': -1.5, ..},
                              or a list of dict-type gate descriptions, e.g. [{'type': 'top', 'k', 2}, {'type': 'top', 'k', 2}],
                              the value of k in top-gating can be also negative, like -2, which indicates one GPU will hold 1/(-k) parameters of an expert
                              capacity_factor X can be positive (factor = X), zero (factor = max(needed_volumes)) or negative (factor = min(-X, max(needed_volumes))).
        model_dim        : the number of channels for MOE's input tensor
        experts          : a dict-type config for builtin expert network
        scan_expert_func : allow users to specify a lambda function to iterate each experts param, e.g. `scan_expert_func = lambda name, param: setattr(param, 'expert', True)`
        result_func      : allow users to specify a lambda function to format the MoE output and aux_loss, e.g. `result_func = lambda output: (output, output.l_aux)`
        group            : specify the explicit communication group of all_to_all
        seeds            : a tuple containing a tripple of int to specify manual seed of (shared params, local params, others params after MoE's)
        a2a_ffn_overlap_degree : the value to control a2a overlap depth, 1 by default for no overlap, 2 for overlap a2a with half gemm, ..
        parallel_type    : the parallel method to compute MoE, valid types: 'auto', 'data', 'model'
        pad_samples      : whether do auto padding on newly-coming input data to maximum data size in history

* Usage of dict-type Experts Config:

        count_per_node   : the number of local experts per device (by default, the value is 1 if not specified)
        type             : available built-in experts implementation, e.g: ffn
        hidden_size_per_expert : the hidden size between two linear layers for each expert (used for type == 'ffn' only)
        activation_fn    : the custom-defined activation function between two linear layers (used for type == 'ffn' only)
        has_fc1_bias     : If set to False, the expert bias parameters `batched_fc1_bias` is disabled. Default: True
        has_fc2_bias     : If set to False, the expert bias parameters `batched_fc2_bias` is disabled. Default: True

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from tutel import moe as tutel_moe

class MoE_Tutel(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts, topk=2):
        super(MoE_Tutel, self).__init__()
        self.num_classes = num_classes
        self.moe_layer = tutel_moe.moe_layer(
            gate_type={'type': 'top', 'k': topk},
            model_dim=input_dim,
            experts={
                'count_per_node': num_experts,
                'type': 'ffn',
                'hidden_size_per_expert': 2048,
                'activation_fn': lambda x: torch.nn.functional.relu(x)
            },
            scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True)
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # 입력 이미지 평탄화
        x = x.view(x.size(0), -1)  # [batch_size, 3072]
        
        output = self.moe_layer(x)
        l_aux = self.moe_layer.l_aux
        return output, l_aux
