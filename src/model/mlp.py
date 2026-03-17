import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w_up = nn.Linear(d_model, d_ff)
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_down(self.w_up(x) * F.silu(self.w_gate(x)))


if __name__ == '__main__':
    model = SwiGLU(128, 128 * 2)
    x = torch.randn(8, 16, 128)
    y = model(x)
    print(y.shape)