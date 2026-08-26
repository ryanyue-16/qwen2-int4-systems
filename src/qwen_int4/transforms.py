"""Function-equivalent parameter transforms used by canonical AWQ."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class Qwen2AWQMapping:
    source: str
    targets: tuple[str, ...]
    transform: str


def qwen2_awq_mappings() -> tuple[Qwen2AWQMapping, ...]:
    """Return the four dense Qwen2 mappings used by the reference implementation."""

    return (
        Qwen2AWQMapping(
            "input_layernorm",
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            "norm_to_linears",
        ),
        Qwen2AWQMapping(
            "self_attn.v_proj", ("self_attn.o_proj",), "gqa_linear_to_linear"
        ),
        Qwen2AWQMapping(
            "post_attention_layernorm",
            ("mlp.gate_proj", "mlp.up_proj"),
            "norm_to_linears",
        ),
        Qwen2AWQMapping("mlp.up_proj", ("mlp.down_proj",), "linear_to_linear"),
    )


def _validated_scales(scales: torch.Tensor, channels: int) -> torch.Tensor:
    values = scales.detach().to(dtype=torch.float32).reshape(-1)
    if values.numel() != channels:
        raise ValueError(f"expected {channels} scales, received {values.numel()}")
    if not torch.isfinite(values).all() or torch.any(values <= 0):
        raise ValueError("AWQ scales must be finite and strictly positive")
    return values


def scale_norm_to_linears(
    norm: nn.Module, linears: Iterable[nn.Linear], scales: torch.Tensor
) -> None:
    """Divide a normalization weight and multiply following weight columns."""

    linears = tuple(linears)
    if not hasattr(norm, "weight") or norm.weight is None:
        raise TypeError("normalization module must expose a weight parameter")
    channels = norm.weight.numel()
    values = _validated_scales(scales, channels)
    for linear in linears:
        if linear.in_features != channels:
            raise ValueError("following linear input size does not match normalization size")
    with torch.no_grad():
        norm.weight.div_(values.to(device=norm.weight.device, dtype=norm.weight.dtype))
        for linear in linears:
            linear.weight.mul_(
                values.to(device=linear.weight.device, dtype=linear.weight.dtype).view(1, -1)
            )


def expand_qwen2_gqa_scales(
    scales: torch.Tensor, *, num_attention_heads: int, num_key_value_heads: int
) -> torch.Tensor:
    """Expand V-projection channel scales to Qwen2's repeated GQA head layout."""

    if num_attention_heads % num_key_value_heads:
        raise ValueError("attention heads must be divisible by key/value heads")
    values = scales.reshape(num_key_value_heads, -1)
    return values.repeat_interleave(
        num_attention_heads // num_key_value_heads, dim=0
    ).reshape(-1)


def scale_linear_to_linear(
    source: nn.Linear,
    target: nn.Linear,
    scales: torch.Tensor,
    *,
    expanded_target_scales: torch.Tensor | None = None,
) -> None:
    """Scale a source output down and the consuming target columns up."""

    values = _validated_scales(scales, source.out_features)
    target_values = values if expanded_target_scales is None else _validated_scales(
        expanded_target_scales, target.in_features
    )
    if target_values.numel() != target.in_features:
        raise ValueError("target scale count does not match target input size")
    with torch.no_grad():
        source.weight.div_(
            values.to(device=source.weight.device, dtype=source.weight.dtype).view(-1, 1)
        )
        if source.bias is not None:
            source.bias.div_(values.to(device=source.bias.device, dtype=source.bias.dtype))
        target.weight.mul_(
            target_values.to(device=target.weight.device, dtype=target.weight.dtype).view(1, -1)
        )


class ModuleStateSnapshot(AbstractContextManager["ModuleStateSnapshot"]):
    """Restore parameters and buffers after a search candidate, including failures."""

    def __init__(self, modules: Iterable[nn.Module]) -> None:
        self.modules = tuple(modules)
        self._states: list[dict[str, torch.Tensor]] = []

    def __enter__(self) -> "ModuleStateSnapshot":
        self._states = [
            {name: value.detach().clone() for name, value in module.state_dict().items()}
            for module in self.modules
        ]
        return self

    def __exit__(self, *_args) -> None:
        for module, state in zip(self.modules, self._states):
            module.load_state_dict(state, strict=True)
        self._states.clear()
