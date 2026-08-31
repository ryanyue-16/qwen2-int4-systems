"""CPU-testable activation-aware scale search for canonical AWQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AWQScaleSearchResult:
    scales: torch.Tensor
    ratio: float
    loss: float
    identity_loss: float


@dataclass(frozen=True)
class AWQClipSearchResult:
    clip_max: torch.Tensor
    ratios: torch.Tensor
    loss: torch.Tensor


def fake_quantize_asymmetric(
    weight: torch.Tensor, *, bits: int = 4, group_size: int = 128
) -> torch.Tensor:
    """Return a dequantized asymmetric group-wise RTN weight for AWQ search."""

    if weight.ndim != 2:
        raise ValueError("weight must be two-dimensional")
    if bits != 4:
        raise ValueError("canonical AWQ currently freezes four-bit weights")
    if weight.shape[1] % group_size:
        raise ValueError("weight input dimension must be divisible by group size")
    grouped = weight.to(torch.float32).reshape(-1, group_size)
    zeros = torch.zeros_like(grouped[:, :1])
    minimum = torch.minimum(grouped.amin(dim=1, keepdim=True), zeros)
    maximum = torch.maximum(grouped.amax(dim=1, keepdim=True), zeros)
    scale = (maximum - minimum).clamp_min(1e-5) / (2**bits - 1)
    zero = torch.round(-minimum / scale).clamp_(0, 2**bits - 1)
    codes = torch.round(grouped / scale + zero).clamp_(0, 2**bits - 1)
    return ((codes - zero) * scale).reshape_as(weight)


def _candidate_scales(
    activation_mean: torch.Tensor,
    weight_mean: torch.Tensor,
    ratio: float,
    *,
    duo_scaling: bool,
) -> torch.Tensor:
    numerator = activation_mean.clamp_min(1e-6).pow(ratio)
    values = (
        numerator / weight_mean.clamp_min(1e-6).pow(1.0 - ratio)
        if duo_scaling
        else numerator
    )
    values = values.clamp_min(1e-4)
    return values / torch.sqrt(values.max() * values.min())


def search_awq_scales(
    inputs: torch.Tensor,
    linears: Iterable[nn.Linear],
    *,
    group_size: int = 128,
    grid_points: int = 20,
    duo_scaling: bool = True,
) -> AWQScaleSearchResult:
    """Search channel scales without mutating any candidate module parameters."""

    linears = tuple(linears)
    if not linears:
        raise ValueError("at least one linear module is required")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    search_device = inputs.device
    flat = inputs.detach().to(device=search_device, dtype=torch.float32).reshape(
        -1, inputs.shape[-1]
    )
    in_features = flat.shape[-1]
    if in_features % group_size:
        raise ValueError("input channels must be divisible by group size")
    if any(linear.in_features != in_features for linear in linears):
        raise ValueError("all searched linears must consume the same input channels")

    weights = [
        linear.weight.detach().to(device=search_device, dtype=torch.float32)
        for linear in linears
    ]
    biases = [
        None
        if linear.bias is None
        else linear.bias.detach().to(device=search_device, dtype=torch.float32)
        for linear in linears
    ]
    references = [F.linear(flat, weight, bias) for weight, bias in zip(weights, biases)]
    activation_mean = flat.abs().mean(dim=0)
    weight_mean = torch.cat(weights, dim=0).abs().mean(dim=0)

    total_squared_error = 0.0
    total_elements = 0
    for weight, bias, reference in zip(weights, biases, references):
        candidate = F.linear(
            flat, fake_quantize_asymmetric(weight, group_size=group_size), bias
        )
        total_squared_error += (candidate - reference).square().sum().item()
        total_elements += reference.numel()
    identity_loss = total_squared_error / total_elements
    best_loss = identity_loss
    best_ratio = 0.0
    best_scales = torch.ones(in_features)
    for index in range(grid_points):
        ratio = index / (grid_points - 1)
        scales = _candidate_scales(
            activation_mean, weight_mean, ratio, duo_scaling=duo_scaling
        )
        scaled_inputs = flat / scales.view(1, -1)
        total_squared_error = 0.0
        total_elements = 0
        for weight, bias, reference in zip(weights, biases, references):
            quantized = fake_quantize_asymmetric(
                weight * scales.view(1, -1), group_size=group_size
            )
            candidate = F.linear(scaled_inputs, quantized, bias)
            total_squared_error += (candidate - reference).square().sum().item()
            total_elements += reference.numel()
        loss = total_squared_error / total_elements
        if loss < best_loss:
            best_loss = loss
            best_ratio = ratio
            best_scales = scales.clone()

    return AWQScaleSearchResult(
        scales=best_scales,
        ratio=best_ratio,
        loss=best_loss,
        identity_loss=identity_loss,
    )


@torch.no_grad()
def search_awq_scales_by_module_output(
    input_batches: Iterable[torch.Tensor],
    linears: Iterable[nn.Linear],
    forward_batch: Callable[[int, torch.Tensor], torch.Tensor],
    *,
    group_size: int = 128,
    grid_points: int = 20,
    duo_scaling: bool = True,
) -> AWQScaleSearchResult:
    """Search input-channel scales against a containing module's output error."""

    batches = tuple(input_batches)
    linears = tuple(linears)
    if not batches:
        raise ValueError("at least one input batch is required")
    if not linears:
        raise ValueError("at least one linear module is required")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    search_device = batches[0].device
    if any(batch.device != search_device for batch in batches):
        raise ValueError("all input batches must use the same device")
    in_features = batches[0].shape[-1]
    if in_features % group_size:
        raise ValueError("input channels must be divisible by group size")
    if any(batch.shape[-1] != in_features for batch in batches):
        raise ValueError("all input batches must have the same channel dimension")
    if any(linear.in_features != in_features for linear in linears):
        raise ValueError("all searched linears must consume the same input channels")

    weights = [linear.weight.detach().clone() for linear in linears]
    flat = torch.cat(
        [batch.detach().to(torch.float32).reshape(-1, in_features) for batch in batches],
        dim=0,
    )
    activation_mean = flat.abs().mean(dim=0)
    weight_mean = torch.cat(
        [weight.to(device=search_device, dtype=torch.float32) for weight in weights],
        dim=0,
    ).abs().mean(dim=0)
    references = [forward_batch(index, batch).detach().to(torch.float32) for index, batch in enumerate(batches)]

    def evaluate(scales: torch.Tensor) -> float:
        for linear, weight in zip(linears, weights):
            candidate = fake_quantize_asymmetric(
                weight.to(device=search_device, dtype=torch.float32)
                * scales.view(1, -1),
                group_size=group_size,
            ) / scales.view(1, -1)
            linear.weight.copy_(candidate.to(dtype=linear.weight.dtype))
        squared_error = 0.0
        elements = 0
        for index, (batch, reference) in enumerate(zip(batches, references)):
            candidate_output = forward_batch(index, batch).to(torch.float32)
            squared_error += (candidate_output - reference).square().sum().item()
            elements += reference.numel()
        return squared_error / elements

    try:
        identity_scales = torch.ones(
            in_features, device=search_device, dtype=torch.float32
        )
        identity_loss = evaluate(identity_scales)
        best_loss = identity_loss
        best_ratio = 0.0
        best_scales = identity_scales.clone()
        for index in range(grid_points):
            ratio = index / (grid_points - 1)
            scales = _candidate_scales(
                activation_mean, weight_mean, ratio, duo_scaling=duo_scaling
            )
            loss = evaluate(scales)
            if loss < best_loss:
                best_loss = loss
                best_ratio = ratio
                best_scales = scales.clone()
    finally:
        for linear, weight in zip(linears, weights):
            linear.weight.copy_(weight)

    return AWQScaleSearchResult(
        scales=best_scales,
        ratio=best_ratio,
        loss=best_loss,
        identity_loss=identity_loss,
    )


def search_gqa_v_to_o_scales(
    inputs: torch.Tensor,
    output_projection: nn.Linear,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    group_size: int = 128,
    grid_points: int = 20,
    duo_scaling: bool = True,
) -> AWQScaleSearchResult:
    """Search V-to-O scales constrained to Qwen2's repeated GQA head layout."""

    if num_attention_heads % num_key_value_heads:
        raise ValueError("attention heads must be divisible by key/value heads")
    search_device = inputs.device
    flat = inputs.detach().to(device=search_device, dtype=torch.float32).reshape(
        -1, inputs.shape[-1]
    )
    if flat.shape[1] != output_projection.in_features:
        raise ValueError("GQA inputs must match the output projection input size")
    if output_projection.in_features % num_attention_heads:
        raise ValueError("output projection input size must divide attention heads")
    if output_projection.in_features % group_size:
        raise ValueError("output projection input size must divide group size")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")

    repeats = num_attention_heads // num_key_value_heads
    head_dim = output_projection.in_features // num_attention_heads
    base_channels = num_key_value_heads * head_dim
    weight = output_projection.weight.detach().to(
        device=search_device, dtype=torch.float32
    )
    bias = (
        None
        if output_projection.bias is None
        else output_projection.bias.detach().to(
            device=search_device, dtype=torch.float32
        )
    )
    reference = F.linear(flat, weight, bias)
    activation_mean = flat.abs().reshape(-1, num_key_value_heads, repeats, head_dim)
    activation_mean = activation_mean.mean(dim=(0, 2)).reshape(base_channels)
    weight_mean = weight.abs().mean(dim=0).reshape(
        num_key_value_heads, repeats, head_dim
    )
    weight_mean = weight_mean.mean(dim=1).reshape(base_channels)

    def expand(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(num_key_value_heads, head_dim).repeat_interleave(
            repeats, dim=0
        ).reshape(-1)

    identity = F.linear(
        flat, fake_quantize_asymmetric(weight, group_size=group_size), bias
    )
    identity_loss = (identity - reference).square().mean().item()
    best_loss = identity_loss
    best_ratio = 0.0
    best_scales = torch.ones(base_channels)
    for index in range(grid_points):
        ratio = index / (grid_points - 1)
        scales = _candidate_scales(
            activation_mean, weight_mean, ratio, duo_scaling=duo_scaling
        )
        expanded = expand(scales)
        candidate = F.linear(
            flat / expanded.view(1, -1),
            fake_quantize_asymmetric(weight * expanded.view(1, -1), group_size=group_size),
            bias,
        )
        loss = (candidate - reference).square().mean().item()
        if loss < best_loss:
            best_loss = loss
            best_ratio = ratio
            best_scales = scales.clone()
    return AWQScaleSearchResult(best_scales, best_ratio, best_loss, identity_loss)


def search_groupwise_clipping(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    *,
    group_size: int = 128,
    ratios: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8),
) -> AWQClipSearchResult:
    """Search per-output, per-group clipping with activation-weighted error."""

    if weight.ndim != 2 or inputs.shape[-1] != weight.shape[1]:
        raise ValueError("weight and activation input dimensions do not match")
    if weight.shape[1] % group_size:
        raise ValueError("weight input dimension must be divisible by group size")
    if not ratios or any(ratio <= 0 or ratio > 1 for ratio in ratios):
        raise ValueError("clip ratios must be in (0, 1]")

    out_features, in_features = weight.shape
    groups = in_features // group_size
    search_device = inputs.device
    grouped = weight.detach().to(device=search_device, dtype=torch.float32).reshape(
        out_features, groups, group_size
    )
    flat_inputs = inputs.detach().to(device=search_device, dtype=torch.float32).reshape(
        -1, in_features
    )
    importance = flat_inputs.square().mean(dim=0).clamp_min(1e-12)
    importance = importance.reshape(1, groups, group_size)
    maximum = grouped.abs().amax(dim=-1).clamp_min(1e-8)
    best_loss = torch.full_like(maximum, float("inf"))
    best_ratio = torch.ones_like(maximum)
    best_max = maximum.clone()

    for ratio in ratios:
        limit = maximum * ratio
        clipped = torch.maximum(
            torch.minimum(grouped, limit.unsqueeze(-1)), -limit.unsqueeze(-1)
        )
        restored = fake_quantize_asymmetric(
            clipped.reshape(out_features, in_features), group_size=group_size
        ).reshape_as(grouped)
        loss = ((grouped - restored).square() * importance).mean(dim=-1)
        improved = loss < best_loss
        best_loss = torch.where(improved, loss, best_loss)
        best_ratio = torch.where(
            improved, torch.full_like(best_ratio, ratio), best_ratio
        )
        best_max = torch.where(improved, limit, best_max)
    return AWQClipSearchResult(best_max, best_ratio, best_loss)
