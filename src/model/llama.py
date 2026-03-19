import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional

from src.model.embedding import Embedding
from src.model.attention import Attention
from src.model.mlp import SwiGLU
from src.model.rmsnorm import RMSNorm
from src.model.rope import RoPE
from src.config.LLaMAConfig import LLaMAConfig
from src.inference.KVCache import KVCache

class LLaMABlock(nn.Module):
    def __init__(self, config: LLaMAConfig, layer_idx: int):
        super(LLaMABlock, self).__init__()
        self.config = config

        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = Attention(config.d_model,
                                   config.num_heads,
                                   config.num_kv_heads,
                                   config.max_seq_len,
                                   RoPE(config.d_model, config.max_seq_len, config.d_model // config.num_heads),
                                   layer_idx)

        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor, start_pos = 0, kv_cache: None | KVCache=None) -> torch.Tensor:
        _x = x
        _x = self.attn_norm(_x)
        _x = self.attention(_x, start_pos = start_pos, kv_cache = kv_cache)
        x = _x + x

        _x = x
        _x = self.ffn_norm(_x)
        _x = self.ffn(_x)
        x = _x + x

        return x

class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super(LLaMA, self).__init__()
        self.config = config
        self.kv_cache = KVCache(
            config.num_layers,
            config.batch_size,
            config.max_seq_len,
            config.num_kv_heads,
            config.d_model // config.num_heads,
            device=config.device
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.layers = nn.ModuleList(
            [LLaMABlock(config, i) for i in range(config.num_layers)]
        )

        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: None | torch.Tensor = None, start_pos: int = 0, use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, seq_len = input_ids.size()

        x = self.embed_tokens(input_ids)

        kv_cache = self.kv_cache if use_cache else None
        for layer in self.layers:
            x = layer(x, start_pos = start_pos, kv_cache = kv_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_token: int, temperature: float, top_k: int, top_p: float, eos_token_id: int | None = None) -> torch.Tensor:
        self.eval()
        bs, seq_len = input_ids.size()

        if self.kv_cache is not None:
            self.kv_cache = KVCache(
            self.config.num_layers,
            self.config.batch_size,
            self.config.max_seq_len,
            self.config.num_kv_heads,
            self.config.d_model // self.config.num_heads,
            device=self.config.device
        )
        # prefill 阶段
        logits, _ = self.forward(input_ids, start_pos=0, use_cache=True)

        next_logits = logits[:, -1, :] / temperature

        if temperature > 0.0:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        generated = torch.cat([input_ids, next_token], dim=1)

        for pos in range(seq_len, seq_len + max_new_token - 1):

            x = generated[:, -1]

            logits, _ = self.forward(input_ids, start_pos=pos, use_cache=True)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0.0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('inf')

            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_prob = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_prob > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = -float('inf')

            if temperature > 0.0:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated



