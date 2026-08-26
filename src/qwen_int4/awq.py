"""CPU-testable activation-aware scale search for canonical AWQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    flat = inputs.detach().to(device="cpu", dtype=torch.float32).reshape(
        -1, inputs.shape[-1]
    )
    in_features = flat.shape[-1]
    if in_features % group_size:
        raise ValueError("input channels must be divisible by group size")
    if any(linear.in_features != in_features for linear in linears):
        raise ValueError("all searched linears must consume the same input channels")

    weights = [linear.weight.detach().to(device="cpu", dtype=torch.float32) for linear in linears]
    biases = [
        None
        if linear.bias is None
        else linear.bias.detach().to(device="cpu", dtype=torch.float32)
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
    grouped = weight.detach().to(device="cpu", dtype=torch.float32).reshape(
        out_features, groups, group_size
    )
    flat_inputs = inputs.detach().to(device="cpu", dtype=torch.float32).reshape(
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
