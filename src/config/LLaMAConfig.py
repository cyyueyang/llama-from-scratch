from dataclasses import dataclass, field
import torch

@dataclass
class LLaMAConfig:
    d_model: int = 2048
    d_ff: int = 2048 * 3
    num_heads: int = 32
    num_kv_heads: int = 4
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    base: float = 10000.0
    vocab_size: int = 50000
    num_layers: int = 16
    batch_size: int = 64
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")