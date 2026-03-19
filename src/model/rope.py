import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000.0):
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        freqs = 1.0 / (self.base ** (torch.arange(0, self.d_model, 2, dtype=torch.float) / self.d_model))
        t = torch.arange(0, self.max_seq_len, dtype=torch.float)

        freqs = torch.outer(t, freqs)

        cos_cached = torch.cos(freqs)
        sin_cached = torch.sin(freqs)
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    def forward(self, x, start_pos=0):
        bs, num_heads, seq_len, d_model = x.size()

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        cos = self.cos_cached[start_pos: start_pos + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[start_pos: start_pos + seq_len].unsqueeze(0).unsqueeze(0)

        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

        return x_rotated



if __name__ == '__main__':
    x = torch.randn(8, 4, 128, 512)
    rope = RoPE(d_model=512, max_seq_len=512)
    print(rope(x).shape)
