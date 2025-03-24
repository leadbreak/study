import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = int(self.expand * self.d_model)

        # Input projection (Linear transformation)
        self.in_proj = nn.Linear(self.d_model, self.inner_dim * 2, bias=False)

        # Convolutional layer
        self.conv = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
        )

        # SSM parameters (discretization)
        self.A = nn.Parameter(torch.randn(self.inner_dim, self.d_state))
        self.B = nn.Parameter(torch.randn(self.inner_dim, self.d_state))
        self.C = nn.Parameter(torch.randn(self.inner_dim, self.d_state))
        self.D = nn.Parameter(torch.ones(self.inner_dim))  # Simplification: D is a learned parameter

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, self.d_model, bias=False)

    def forward(self, x):
        """
        x (torch.Tensor): (B, L, D)
        """
        B, L, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # (B, L, inner_dim * 2)
        x_proj = x_proj.view(B, L, self.inner_dim, 2)  # (B, L, inner_dim, 2)
        x, gate = x_proj.split(1, dim=-1)  # (B, L, inner_dim, 1), (B, L, inner_dim, 1)
        x = x.squeeze(-1)  # (B, L, inner_dim)
        gate = gate.squeeze(-1)  # (B, L, inner_dim)

        # Convolution
        x = x.transpose(1, 2)  # (B, inner_dim, L)
        x = self.conv(x)[:, :, :L]  # (B, inner_dim, L) # Truncate padding
        x = x.transpose(1, 2)  # (B, L, inner_dim)

        # SSM
        delta = F.sigmoid(x)  # (B, L, inner_dim)
        A = self.A * delta.unsqueeze(-1)  # (B, L, inner_dim, d_state)
        B = self.B * delta.unsqueeze(-1)  # (B, L, inner_dim, d_state)
        C = self.C  # (inner_dim, d_state)
        D = self.D  # (inner_dim)

        # State space model
        state = torch.zeros(B, self.inner_dim, self.d_state, device=x.device)  # (B, inner_dim, d_state)
        output = []
        for l in range(L):
            state = A[:, l] @ state + B[:, l] * x[:, l].unsqueeze(-1)  # (B, inner_dim, d_state)
            output_l = (state @ C.unsqueeze(0)).squeeze(-1)  # (B, inner_dim)
            output.append(output_l)
        output = torch.stack(output, dim=1)  # (B, L, inner_dim)

        # Gate and output projection
        output = output * F.sigmoid(gate)  # (B, L, inner_dim)
        output = self.out_proj(output)  # (B, L, D)

        return output

if __name__ == '__main__':
    
    # Example usage
    batch_size = 2
    seq_len = 10
    d_model = 64

    # Create a Mamba layer
    model = Mamba(d_model=d_model)
    
    from torchinfo import summary

    # # Describe the model
    # device = 'cuda'
    # summary(model.to(device), input_size=(batch_size, seq_len, d_model), dtypes=[torch.Tensor], device=device)

    # Create a dummy input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Pass the input through the Mamba layer
    output = model(x)

    # Print the output shape
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)