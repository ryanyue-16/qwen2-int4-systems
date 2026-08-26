"""Small, dependency-stable Qwen2 architecture for the local INT4 checkpoint."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .linear import QuantLinear
from .quantization import PackingFormat


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.einsum("bt,d->btd", position_ids.to(torch.float32), self.inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.to(torch.float32) * torch.rsqrt(
            x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized.to(x.dtype) * self.weight).to(x.dtype)


class QwenAttention(nn.Module):
    def __init__(self, config, *, backend: str, packing: PackingFormat, group_size: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        common = {"group_size": group_size, "backend": backend, "packing": packing}
        self.q_proj = QuantLinear(self.hidden_size, self.num_heads * self.head_dim, bias=True, **common)
        self.k_proj = QuantLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True, **common)
        self.v_proj = QuantLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True, **common)
        self.o_proj = QuantLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, **common)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scores = (q.to(torch.float32) @ k.to(torch.float32).transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=hidden_states.device), diagonal=1
        )
        scores.masked_fill_(causal.view(1, 1, sequence, sequence), float("-inf"))
        if attention_mask is not None:
            key_is_padding = ~attention_mask.to(torch.bool).view(batch, 1, 1, sequence)
            scores.masked_fill_(key_is_padding, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output = probabilities @ v.to(torch.float32)
        output = output.transpose(1, 2).contiguous().view(batch, sequence, self.hidden_size)
        return self.o_proj(output.to(hidden_states.dtype))


class QwenMLP(nn.Module):
    def __init__(self, config, *, backend: str, packing: PackingFormat, group_size: int) -> None:
        super().__init__()
        common = {"group_size": group_size, "backend": backend, "packing": packing}
        self.gate_proj = QuantLinear(config.hidden_size, config.intermediate_size, **common)
        self.up_proj = QuantLinear(config.hidden_size, config.intermediate_size, **common)
        self.down_proj = QuantLinear(config.intermediate_size, config.hidden_size, **common)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x).to(torch.float32))
        up = self.up_proj(x).to(torch.float32)
        return self.down_proj(gated * up).to(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, config, *, backend: str, packing: PackingFormat, group_size: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = QwenAttention(
            config, backend=backend, packing=packing, group_size=group_size
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = QwenMLP(config, backend=backend, packing=packing, group_size=group_size)

    def forward(self, x, cos, sin, attention_mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class QwenModel(nn.Module):
    def __init__(self, config, *, backend: str, packing: PackingFormat, group_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
        self.layers = nn.ModuleList(
            DecoderLayer(config, backend=backend, packing=packing, group_size=group_size)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = RotaryEmbedding(head_dim, getattr(config, "rope_theta", 1_000_000.0))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
        position_ids.clamp_min_(0)
        cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        return self.norm(hidden_states)


class QwenInt4ForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        *,
        backend: str = "torch",
        packing: PackingFormat | str = PackingFormat.LEGACY_SIGNED_ADD,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        packing = PackingFormat(packing)
        self.config = config
        self.model = QwenModel(
            config, backend=backend, packing=packing, group_size=group_size
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.model.embed_tokens.weight

    def set_backend(self, backend: str) -> None:
        for module in self.modules():
            if isinstance(module, QuantLinear):
                module.set_backend(backend)

    def set_packing(self, packing: PackingFormat | str) -> None:
        packing = PackingFormat(packing)
        for module in self.modules():
            if isinstance(module, QuantLinear):
                module.packing = packing

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        return self.lm_head(hidden_states)
