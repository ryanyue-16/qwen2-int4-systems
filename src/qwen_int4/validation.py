"""Numerical comparison helpers shared by validation scripts and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from inspect import signature
from typing import Any
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TensorMetrics:
    shape: tuple[int, ...]
    max_abs: float
    mean_abs: float
    rmse: float
    mean_relative: float
    cosine: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def language_model_metrics(logits: torch.Tensor, input_ids: torch.Tensor) -> dict[str, float]:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [B,T,V] and input_ids [B,T]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError("logits/input token shapes must match and contain at least two tokens")
    shift_logits = logits[:, :-1].to(torch.float32)
    labels = input_ids[:, 1:]
    nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), labels.reshape(-1)).item()
    return {"nll": nll, "perplexity": math.exp(min(nll, 20.0))}


def sliding_window_language_model_metrics(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    window_size: int,
    stride: int,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor]:
    """Score a long sequence without retaining vocabulary-sized logits.

    Windows overlap to provide context, while each target token is scored exactly
    once. Only one top-1 ID and one target log-prob are retained per token.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence_length]")
    if input_ids.shape[1] < 2:
        raise ValueError("at least two tokens are required")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if stride < 1 or stride >= window_size:
        raise ValueError("stride must satisfy 1 <= stride < window_size")

    total_nll = 0.0
    predictions: list[torch.Tensor] = []
    target_log_probs: list[torch.Tensor] = []
    sequence_length = input_ids.shape[1]
    num_windows = 0
    supports_use_cache = "use_cache" in signature(model.forward).parameters

    for target_start in range(1, sequence_length, stride):
        target_end = min(target_start + stride, sequence_length)
        context_start = max(0, target_end - window_size)
        window_ids = input_ids[:, context_start:target_end].to(device)
        local_target_start = target_start - context_start
        local_target_end = target_end - context_start

        output = (
            model(window_ids, use_cache=False)
            if supports_use_cache
            else model(window_ids)
        )
        logits = output.logits if hasattr(output, "logits") else output
        scored_logits = logits[
            :, local_target_start - 1 : local_target_end - 1
        ].to(torch.float32)
        targets = window_ids[:, local_target_start:local_target_end]
        flat_logits = scored_logits.reshape(-1, scored_logits.shape[-1])
        flat_targets = targets.reshape(-1)
        losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        log_probs = F.log_softmax(flat_logits, dim=-1)

        total_nll += losses.sum().item()
        predictions.append(flat_logits.argmax(dim=-1).detach().cpu())
        target_log_probs.append(
            log_probs.gather(1, flat_targets[:, None]).squeeze(1).detach().cpu()
        )
        num_windows += 1

    all_predictions = torch.cat(predictions)
    all_target_log_probs = torch.cat(target_log_probs)
    num_predictions = all_predictions.numel()
    nll = total_nll / num_predictions
    metrics: dict[str, float | int] = {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "predictions": num_predictions,
        "windows": num_windows,
        "window_size": window_size,
        "stride": stride,
    }
    return metrics, all_predictions, all_target_log_probs


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> TensorMetrics:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, candidate={candidate.shape}")
    ref = reference.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    got = candidate.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if ref.numel() == 0:
        raise ValueError("cannot compare empty tensors")
    difference = (ref - got).abs()
    relative = difference / ref.abs().clamp_min(1e-6)
    ref_norm = torch.linalg.vector_norm(ref)
    got_norm = torch.linalg.vector_norm(got)
    denominator = ref_norm * got_norm
    cosine = 1.0 if denominator.item() == 0 and torch.equal(ref, got) else (
        0.0 if denominator.item() == 0 else torch.dot(ref, got).div(denominator).item()
    )
    return TensorMetrics(
        shape=tuple(reference.shape),
        max_abs=difference.max().item(),
        mean_abs=difference.mean().item(),
        rmse=torch.sqrt(torch.mean((ref - got).square())).item(),
        mean_relative=relative.mean().item(),
        cosine=cosine,
    )


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if hasattr(value, "logits") and isinstance(value.logits, torch.Tensor):
        return value.logits
    return None


class ActivationRecorder:
    """Forward-hook recorder which stores selected outputs as CPU tensors."""

    def __init__(self, module: torch.nn.Module, names: set[str]) -> None:
        self.activations: dict[str, torch.Tensor] = {}
        self._handles = []
        available = dict(module.named_modules())
        missing = names - set(available)
        if missing:
            raise KeyError(f"modules not found: {sorted(missing)}")
        for name in names:
            self._handles.append(available[name].register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def save(_module, _inputs, output) -> None:
            tensor = first_tensor(output)
            if tensor is not None:
                self.activations[name] = tensor.detach().to("cpu")

        return save

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivationRecorder":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def qwen_module_names(num_layers: int) -> list[str]:
    names = ["model.embed_tokens"]
    for index in range(num_layers):
        prefix = f"model.layers.{index}"
        names.extend(
            [
                f"{prefix}.input_layernorm",
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.post_attention_layernorm",
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
                f"{prefix}.mlp.down_proj",
                prefix,
            ]
        )
    names.extend(["model.norm", "lm_head"])
    return names
