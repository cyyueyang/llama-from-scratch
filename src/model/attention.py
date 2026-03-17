import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.model.rope import RoPE

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len, rope):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.rope = rope

        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.group = num_heads // num_kv_heads
        self.w_qkv = nn.Linear(d_model, self.head_dim * (self.num_heads + 2 * self.num_kv_heads))

        self.w_o = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x):
        bs, seq_len, d_model = x.shape
        q, k, v = self.w_qkv(x).split([self.head_dim * self.num_heads, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], dim=-1)

        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = self.repeat(k, self.group)
        v = self.repeat(v, self.group)

        q = self.rope(q)
        k = self.rope(k)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask.unsqueeze(0).unsqueeze(0)[:, :, :seq_len, :seq_len], -1e20)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        out = self.w_o(out)

        return out

    def repeat(self, x, group):
        bs, num_heads, seq_len, head_dim = x.size()

        x = x.repeat(1, group, 1, 1)

        return x

if __name__ == '__main__':
    x = torch.randn(8, 128, 512)
    rope = RoPE(32)
    attention = Attention(512, 16, 8, 2048, rope)
    print(attention(x).shape)


