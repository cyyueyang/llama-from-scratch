import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple, Optional

class KVCache:
    def __init__(self, num_layers: int, max_batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int, device: torch.device) -> None:
        self.num_layers: int = num_layers
        self.max_batch_size: int = max_batch_size
        self.max_seq_len: int = max_seq_len
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim

        self.k_cache = torch.zeros(
            self.num_layers, self.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
            dtype=torch.float,
            device=device
        )

        self.v_cache = torch.zeros(
            self.num_layers, self.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
            dtype=torch.float,
            device=device
        )

    def update(self, layer_idx: int, start_pos: int, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, num_kv_heads, seq_len, head_dim = xk.shape

        self.k_cache[layer_idx, :bs, :, start_pos: start_pos + seq_len, :] = xk
        self.v_cache[layer_idx, :bs, :, start_pos: start_pos + seq_len, :] = xv

        keys = self.k_cache[layer_idx, :bs, :, :start_pos + seq_len, :]
        values = self.v_cache[layer_idx, :bs, :, :start_pos + seq_len, :]

        return keys, values