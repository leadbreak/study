import torch
import torch.nn as nn
from fairscale.nn.moe import Top2Gate, MOELayer

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.layer(x)

gate = Top2Gate(model_dim, num_experts)
moe = MOELayer(gate, expert)
output = moe(input)