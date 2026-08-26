"""INT4 storage-format primitives used by every backend.

The local checkpoint was produced by an old packer which appears to have added
a signed low nibble directly to ``high << 4``.  A negative low value borrows
from the stored high nibble.  ``LEGACY_SIGNED_ADD`` reverses that borrow before
sign extension.  Keeping this rule here makes the checkpoint contract explicit
and prevents each backend from inventing its own decoder.
"""

from __future__ import annotations

from enum import Enum

import torch


class PackingFormat(str, Enum):
    # Canonical v1: byte = q[k] in low nibble | q[k+1] in high nibble.
    CANONICAL = "canonical"
    LEGACY_SIGNED_ADD = "legacy_signed_add"


def _validate_codes(codes: torch.Tensor) -> None:
    if codes.numel() % 2:
        raise ValueError("INT4 packing requires an even number of elements")
    if codes.numel() and (codes.min().item() < -8 or codes.max().item() > 7):
        raise ValueError("signed INT4 codes must be in [-8, 7]")


def pack_int4(
    codes: torch.Tensor,
    *,
    packing: PackingFormat | str = PackingFormat.CANONICAL,
) -> torch.Tensor:
    """Pack consecutive signed INT4 values into uint8 bytes.

    Canonical v1 stores the first/K-even value in the low nibble and the
    second/K-odd value in the high nibble. The legacy format retains the old
    signed-add behavior and its historical high/low ordering.
    """
    packing = PackingFormat(packing)
    flat = codes.reshape(-1).to(torch.int16)
    _validate_codes(flat)
    first, second = flat[0::2], flat[1::2]

    if packing is PackingFormat.LEGACY_SIGNED_ADD:
        packed = (first << 4) + second
    else:
        packed = (first & 0x0F) | ((second & 0x0F) << 4)
    return (packed & 0xFF).to(torch.uint8)


def unpack_int4(
    packed: torch.Tensor,
    *,
    packing: PackingFormat | str = PackingFormat.CANONICAL,
    elements: int | None = None,
) -> torch.Tensor:
    """Decode packed bytes to an interleaved int8 tensor in signed INT4 range."""
    packing = PackingFormat(packing)
    byte = packed.reshape(-1).to(torch.uint8)
    low_u = (byte & 0x0F).to(torch.int16)
    high_u = (byte >> 4).to(torch.int16)

    if packing is PackingFormat.LEGACY_SIGNED_ADD:
        # The old packer used `(high << 4) + low` while low was signed.
        # Reverse the borrow caused by a negative low nibble.
        high_u = (high_u + (low_u >= 8).to(torch.int16)) & 0x0F

    low = torch.where(low_u >= 8, low_u - 16, low_u)
    high = torch.where(high_u >= 8, high_u - 16, high_u)
    result = torch.empty(byte.numel() * 2, dtype=torch.int8, device=byte.device)
    if packing is PackingFormat.LEGACY_SIGNED_ADD:
        result[0::2] = high.to(torch.int8)
        result[1::2] = low.to(torch.int8)
    else:
        result[0::2] = low.to(torch.int8)
        result[1::2] = high.to(torch.int8)

    if elements is not None:
        if elements < 0 or elements > result.numel():
            raise ValueError(f"invalid element count {elements} for {result.numel()} decoded values")
        result = result[:elements]
    return result


def quantize_symmetric_rtn(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
    packing: PackingFormat | str = PackingFormat.CANONICAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight with per-output, K-group symmetric RTN W4.

    The range is deliberately ``[-7, 7]`` so positive and negative magnitude
    are symmetric. Scales are returned as FP16; packed weights are uint8.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f"in_features={in_features} is not divisible by group_size={group_size}")
    groups = in_features // group_size
    grouped = weight.detach().to(device="cpu", dtype=torch.float32).reshape(
        out_features, groups, group_size
    )
    scales_fp32 = grouped.abs().amax(dim=-1).div(7.0)
    scales_fp32 = torch.where(scales_fp32 == 0, torch.ones_like(scales_fp32), scales_fp32)
    codes = torch.round(grouped / scales_fp32.unsqueeze(-1)).clamp_(-7, 7).to(torch.int8)
    packed = pack_int4(codes, packing=packing).reshape(-1, 1)
    return packed, scales_fp32.to(torch.float16)


def quantize_activation_weighted_clip(
    weight: torch.Tensor,
    activation_importance: torch.Tensor,
    *,
    group_size: int = 128,
    clip_ratios: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7),
    packing: PackingFormat | str = PackingFormat.CANONICAL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activation-weighted per-group clipping search followed by symmetric RTN.

    The diagonal activation second moment approximates output reconstruction
    error: ``E[(x @ (W-Wq).T)^2]``. This is a transparent calibration method,
    not a full implementation of AWQ channel scaling or GPTQ compensation.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if activation_importance.shape != (in_features,):
        raise ValueError(
            f"activation importance must have shape {(in_features,)}, "
            f"got {tuple(activation_importance.shape)}"
        )
    if in_features % group_size:
        raise ValueError(f"in_features={in_features} is not divisible by group_size={group_size}")
    if not clip_ratios or any(ratio <= 0 or ratio > 1 for ratio in clip_ratios):
        raise ValueError("clip ratios must be in (0, 1]")

    groups = in_features // group_size
    grouped = weight.detach().to(device="cpu", dtype=torch.float32).reshape(
        out_features, groups, group_size
    )
    importance = activation_importance.detach().to(device="cpu", dtype=torch.float32)
    importance = importance.clamp_min(1e-12).reshape(1, groups, group_size)
    importance = importance / importance.mean(dim=-1, keepdim=True)
    maximum = grouped.abs().amax(dim=-1)
    best_loss = torch.full_like(maximum, float("inf"))
    best_scale = torch.ones_like(maximum)
    best_ratio = torch.ones_like(maximum)
    best_codes = torch.zeros_like(grouped, dtype=torch.int8)

    for ratio in clip_ratios:
        scale = maximum.mul(ratio).div(7.0)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = torch.round(grouped / scale.unsqueeze(-1)).clamp_(-7, 7).to(torch.int8)
        restored = codes.to(torch.float32) * scale.unsqueeze(-1)
        loss = ((grouped - restored).square() * importance).mean(dim=-1)
        improved = loss < best_loss
        best_loss = torch.where(improved, loss, best_loss)
        best_scale = torch.where(improved, scale, best_scale)
        best_ratio = torch.where(improved, torch.full_like(best_ratio, ratio), best_ratio)
        best_codes = torch.where(improved.unsqueeze(-1), codes, best_codes)

    packed = pack_int4(best_codes, packing=packing).reshape(-1, 1)
    return packed, best_scale.to(torch.float16), best_ratio.to(torch.float16)


def dequantize_groupwise(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    packing: PackingFormat | str = PackingFormat.LEGACY_SIGNED_ADD,
    zeros: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference symmetric group-wise INT4 -> FP32 dequantization."""
    if in_features % group_size:
        raise ValueError(f"in_features={in_features} is not divisible by group_size={group_size}")
    groups = in_features // group_size
    if tuple(scales.shape) != (out_features, groups):
        raise ValueError(
            f"scale shape must be {(out_features, groups)}, got {tuple(scales.shape)}"
        )
    expected_bytes = out_features * in_features // 2
    if qweight.numel() != expected_bytes:
        raise ValueError(f"qweight must contain {expected_bytes} bytes, got {qweight.numel()}")
    if zeros is not None and torch.count_nonzero(zeros).item() != 0:
        raise NotImplementedError("this reference currently supports symmetric checkpoints only")

    codes = unpack_int4(qweight, packing=packing).reshape(out_features, in_features)
    expanded_scales = scales.to(torch.float32).repeat_interleave(group_size, dim=1)
    return codes.to(torch.float32) * expanded_scales
