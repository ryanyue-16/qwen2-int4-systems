"""Memory-bounded activation collection for sequential AWQ calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class ActivationStatistics:
    count: int
    mean_abs: torch.Tensor
    second_moment: torch.Tensor


class ActivationCollector:
    """Collect per-input-channel statistics and a bounded token sample cache."""

    def __init__(self, *, max_cached_tokens: int = 512) -> None:
        if max_cached_tokens < 1:
            raise ValueError("max_cached_tokens must be positive")
        self.max_cached_tokens = max_cached_tokens
        self._count: dict[str, int] = {}
        self._sum_abs: dict[str, torch.Tensor] = {}
        self._sum_sq: dict[str, torch.Tensor] = {}
        self._samples: dict[str, list[torch.Tensor]] = {}
        self._sample_counts: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str):
        def collect(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output) -> None:
            if not inputs:
                raise RuntimeError(f"module {name} received no positional activation")
            values = inputs[0].detach().to(device="cpu", dtype=torch.float32)
            flat = values.reshape(-1, values.shape[-1])
            self._count[name] = self._count.get(name, 0) + flat.shape[0]
            absolute = flat.abs().sum(dim=0)
            squared = flat.square().sum(dim=0)
            self._sum_abs[name] = self._sum_abs.get(
                name, torch.zeros_like(absolute)
            ) + absolute
            self._sum_sq[name] = self._sum_sq.get(
                name, torch.zeros_like(squared)
            ) + squared

            remaining = self.max_cached_tokens - self._sample_counts.get(name, 0)
            if remaining > 0:
                cached = flat[:remaining].clone()
                self._samples.setdefault(name, []).append(cached)
                self._sample_counts[name] = self._sample_counts.get(name, 0) + len(cached)

        return collect

    def attach(self, module: nn.Module, names: Iterable[str] | None = None) -> None:
        if self._handles:
            raise RuntimeError("collector is already attached")
        selected = set(names) if names is not None else None
        available = dict(module.named_modules())
        if selected is not None:
            missing = selected - set(available)
            if missing:
                raise KeyError(f"calibration modules not found: {sorted(missing)}")
        for name, child in available.items():
            if isinstance(child, nn.Linear) and (selected is None or name in selected):
                self._handles.append(child.register_forward_hook(self._hook(name)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def statistics(self) -> dict[str, ActivationStatistics]:
        return {
            name: ActivationStatistics(
                count=count,
                mean_abs=self._sum_abs[name] / count,
                second_moment=self._sum_sq[name] / count,
            )
            for name, count in self._count.items()
        }

    def samples(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.cat(chunks, dim=0)
            for name, chunks in self._samples.items()
            if chunks
        }

    def __enter__(self) -> "ActivationCollector":
        return self

    def __exit__(self, *_args) -> None:
        self.detach()


def sequential_block_calibration(
    blocks: Iterable[nn.Module],
    hidden_states: torch.Tensor,
    *,
    forward_block: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    max_cached_tokens: int = 512,
) -> tuple[list[dict[str, ActivationStatistics]], torch.Tensor]:
    """Calibrate one block at a time and retain only the next block input.

    The callback makes attention masks, rotary inputs, and model-specific output
    structures explicit at the call site. Each block's hooks are removed before
    the next block starts.
    """

    current = hidden_states.detach()
    per_block: list[dict[str, ActivationStatistics]] = []
    for block in blocks:
        collector = ActivationCollector(max_cached_tokens=max_cached_tokens)
        collector.attach(block)
        try:
            with torch.inference_mode():
                current = forward_block(block, current)
        finally:
            collector.detach()
        if isinstance(current, (tuple, list)):
            current = current[0]
        if not isinstance(current, torch.Tensor):
            raise TypeError("forward_block must return a tensor or a tensor-first tuple")
        per_block.append(collector.statistics())
        current = current.detach()
    return per_block, current
