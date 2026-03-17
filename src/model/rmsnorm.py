import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def normalize(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.normalize(x) * self.weight

if __name__ == '__main__':
    norm = RMSNorm(128)
    x = torch.randn(8, 16, 128)
    y = norm(x)
    print(y.shape)
