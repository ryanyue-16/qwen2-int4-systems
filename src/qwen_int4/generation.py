"""Correctness-first cached generation helpers for Phase F."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CachedGenerationState:
    """Owned KV cache and mask state for one left-padded generation batch."""

    model: torch.nn.Module
    generated_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor
    past_key_values: tuple
    active: bool = True

    def _require_active(self) -> None:
        if not self.active:
            raise RuntimeError("cached generation state is closed")

    @classmethod
    def prefill(
        cls,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> "CachedGenerationState":
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non_empty_sequence]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        mask = attention_mask.to(dtype=torch.bool)
        for row in mask:
            first_valid = int(row.nonzero(as_tuple=False)[0].item()) if row.any() else row.numel()
            if first_valid == row.numel() or not bool(row[first_valid:].all()):
                raise ValueError("cached batching supports only non-empty left-padded attention masks")
        with torch.inference_mode():
            logits, cache = model(input_ids, attention_mask=mask, use_cache=True)
        if not isinstance(cache, tuple):
            raise RuntimeError("cache-enabled model did not return per-layer key/value tensors")
        return cls(model, input_ids, mask, logits, cache)

    def next_greedy_tokens(self) -> torch.Tensor:
        self._require_active()
        return self.logits[:, -1].argmax(dim=-1, keepdim=True)

    def decode(self, next_tokens: torch.Tensor) -> torch.Tensor:
        self._require_active()
        if next_tokens.ndim != 2 or next_tokens.shape != (self.generated_ids.shape[0], 1):
            raise ValueError("next_tokens must have shape [batch, 1]")
        with torch.inference_mode():
            self.generated_ids = torch.cat((self.generated_ids, next_tokens), dim=1)
            self.attention_mask = torch.cat(
                (self.attention_mask, torch.ones_like(next_tokens, dtype=torch.bool)), dim=1
            )
            self.logits, self.past_key_values = self.model(
                next_tokens,
                attention_mask=self.attention_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
        return self.logits

    def select_requests(self, indices: torch.Tensor | list[int]) -> None:
        """Reorder or remove requests while preserving cache ownership."""
        self._require_active()
        selected = torch.as_tensor(indices, dtype=torch.long, device=self.generated_ids.device)
        if selected.ndim != 1 or selected.numel() == 0:
            raise ValueError("request indices must be a non-empty one-dimensional sequence")
        batch = self.generated_ids.shape[0]
        if selected.min().item() < 0 or selected.max().item() >= batch:
            raise IndexError("request index is outside the cached batch")
        if selected.unique().numel() != selected.numel():
            raise ValueError("duplicate request indices require explicit beam-cache semantics")
        self.generated_ids = self.generated_ids.index_select(0, selected)
        self.attention_mask = self.attention_mask.index_select(0, selected)
        self.logits = self.logits.index_select(0, selected)
        self.past_key_values = tuple(
            (key.index_select(0, selected), value.index_select(0, selected))
            for key, value in self.past_key_values
        )

    def close(self) -> None:
        """Release owned cache tensors and make further use fail closed."""
        self.past_key_values = ()
        self.logits = torch.empty(0, device=self.generated_ids.device)
        self.active = False


def greedy_generate_cached(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate greedy continuations using one prefill and cached decode steps.

    All batch members must provide a non-empty left-padded prompt. Explicit
    request selection, reordering, removal, and cache release are available on
    ``CachedGenerationState``; duplicate indices remain reserved for future
    beam-cache semantics.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    state = CachedGenerationState.prefill(model, input_ids, attention_mask)
    for _ in range(max_new_tokens):
        state.decode(state.next_greedy_tokens())
    return state.generated_ids
