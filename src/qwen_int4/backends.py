"""Operator backends for quantized linear layers."""

from __future__ import annotations

import torch

from .quantization import PackingFormat, dequantize_groupwise, unpack_int4
from .triton_kernels import linear_canonical_w4g64_triton, triton_available


SUPPORTED_BACKENDS = ("torch", "lpu", "triton")


def torch_reference_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    packing: PackingFormat,
) -> torch.Tensor:
    """Correctness-first W4A16 path: dequantize the full weight, then matmul."""
    weight = dequantize_groupwise(
        qweight,
        scales,
        zeros=zeros,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        packing=packing,
    )
    output = x.reshape(-1, in_features).to(torch.float32) @ weight.t()
    if bias is not None:
        output = output + bias.to(torch.float32)
    return output.reshape(*x.shape[:-1], out_features).to(x.dtype)


def lpu_functional_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    packing: PackingFormat,
) -> torch.Tensor:
    """K-streaming functional model of the LPU A/B/C-buffer dataflow.

    This is deliberately not a cycle-accurate or bit-accurate hardware model.
    It uses FP32 PyTorch matmul for each K group and FP32 C-buffer accumulation.
    """
    groups = in_features // group_size
    codes = unpack_int4(qweight, packing=packing).reshape(out_features, in_features)
    zero_values = unpack_int4(
        zeros, packing=packing, elements=out_features * groups
    ).reshape(out_features, groups)
    activation = x.reshape(-1, in_features).to(torch.float32)
    accumulator = torch.zeros(
        (activation.shape[0], out_features), device=x.device, dtype=torch.float32
    )

    for group in range(groups):
        start = group * group_size
        stop = start + group_size
        a_tile = activation[:, start:stop]
        b_tile = codes[:, start:stop].to(torch.float32)
        b_tile = b_tile - zero_values[:, group].to(torch.float32).unsqueeze(1)
        b_tile = b_tile * scales[:, group].to(torch.float32).unsqueeze(1)
        accumulator.add_(a_tile @ b_tile.t())

    if bias is not None:
        accumulator.add_(bias.to(torch.float32))
    return accumulator.reshape(*x.shape[:-1], out_features).to(x.dtype)


def triton_w4g64_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    packing: PackingFormat,
) -> torch.Tensor:
    """Dispatch canonical asymmetric W4G64 linear to the Phase E Triton operator."""
    if not triton_available():
        raise RuntimeError("Triton backend requires CUDA and a Triton installation")
    if packing is not PackingFormat.CANONICAL or group_size != 64:
        raise ValueError("Triton backend supports only canonical asymmetric W4G64 tensors")
    if x.shape[-1] != in_features:
        raise ValueError("activation feature dimension does not match QuantLinear")
    if x.dtype == torch.float32:
        return torch_reference_linear(
            x, qweight, scales, zeros, bias,
            out_features=out_features, in_features=in_features,
            group_size=group_size, packing=packing,
        )
    output = linear_canonical_w4g64_triton(
        x.reshape(-1, in_features).contiguous(), qweight, scales, zeros, bias
    )
    return output.reshape(*x.shape[:-1], out_features)
