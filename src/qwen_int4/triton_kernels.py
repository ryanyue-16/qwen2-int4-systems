"""Correctness-first Triton primitives for canonical W4G64 asymmetric AWQ."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CUDA/Triton hosts only.
    triton = None
    tl = None


def triton_available() -> bool:
    """Return whether this process can execute the Phase E Triton primitives."""
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _unpack_canonical_signed_int4_kernel(
        packed_ptr,
        output_ptr,
        elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        packed = tl.load(packed_ptr + offsets // 2, mask=mask, other=0).to(tl.int32)
        nibble_shift = (offsets & 1) * 4
        unsigned = (packed >> nibble_shift) & 0x0F
        signed = tl.where(unsigned >= 8, unsigned - 16, unsigned)
        tl.store(output_ptr + offsets, signed, mask=mask)


    @triton.jit
    def _dequantize_canonical_w4g64_kernel(
        qweight_ptr,
        scales_ptr,
        zeros_ptr,
        output_ptr,
        elements: tl.constexpr,
        in_features: tl.constexpr,
        groups_per_row: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        packed_codes = tl.load(qweight_ptr + offsets // 2, mask=mask, other=0).to(tl.int32)
        code_unsigned = (packed_codes >> ((offsets & 1) * 4)) & 0x0F
        codes = tl.where(code_unsigned >= 8, code_unsigned - 16, code_unsigned)

        rows = offsets // in_features
        groups = (offsets % in_features) // 64
        zero_offsets = rows * groups_per_row + groups
        packed_zeros = tl.load(zeros_ptr + zero_offsets // 2, mask=mask, other=0).to(tl.int32)
        zero_unsigned = (packed_zeros >> ((zero_offsets & 1) * 4)) & 0x0F
        zeros = tl.where(zero_unsigned >= 8, zero_unsigned - 16, zero_unsigned)
        scales = tl.load(scales_ptr + zero_offsets, mask=mask, other=0.0).to(tl.float32)
        output = (codes - zeros).to(tl.float32) * scales
        tl.store(output_ptr + offsets, output, mask=mask)


    @triton.jit
    def _linear_canonical_w4g64_kernel(
        activations_ptr,
        qweight_ptr,
        scales_ptr,
        zeros_ptr,
        bias_ptr,
        output_ptr,
        rows: tl.constexpr,
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        groups_per_row: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_offsets = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
        column_offsets = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for group in tl.range(0, groups_per_row):
            columns = group * BLOCK_K + k_offsets
            activation = tl.load(
                activations_ptr + row_offsets[:, None] * in_features + columns[None, :],
                mask=row_offsets[:, None] < rows,
                other=0.0,
            )
            logical_offsets = column_offsets[:, None] * in_features + columns[None, :]
            column_mask = column_offsets[:, None] < out_features
            packed_codes = tl.load(qweight_ptr + logical_offsets // 2, mask=column_mask, other=0).to(tl.int32)
            code_unsigned = (packed_codes >> ((logical_offsets & 1) * 4)) & 0x0F
            codes = tl.where(code_unsigned >= 8, code_unsigned - 16, code_unsigned)
            zero_offsets = column_offsets * groups_per_row + group
            packed_zeros = tl.load(zeros_ptr + zero_offsets // 2, mask=column_offsets < out_features, other=0).to(tl.int32)
            zero_unsigned = (packed_zeros >> ((zero_offsets & 1) * 4)) & 0x0F
            zero_values = tl.where(zero_unsigned >= 8, zero_unsigned - 16, zero_unsigned)
            scales = tl.load(scales_ptr + zero_offsets, mask=column_offsets < out_features, other=0.0).to(tl.float32)
            weight = (codes - zero_values[:, None]).to(tl.float32) * scales[:, None]
            accumulator += tl.dot(
                activation.to(tl.float32),
                tl.trans(weight),
                input_precision="ieee",
            )
        if HAS_BIAS:
            accumulator += tl.load(bias_ptr + column_offsets, mask=column_offsets < out_features, other=0.0)
        tl.store(
            output_ptr + row_offsets[:, None] * out_features + column_offsets[None, :],
            accumulator,
            mask=(row_offsets[:, None] < rows) & (column_offsets[None, :] < out_features),
        )


def unpack_canonical_signed_int4_triton(
    packed: torch.Tensor, *, elements: int | None = None
) -> torch.Tensor:
    """Decode canonical low-even/high-odd packed signed INT4 values on CUDA.

    ``elements`` may truncate the final byte, as required for an odd count of
    packed asymmetric zero points. The function intentionally supports only the
    frozen canonical format; legacy signed-add bytes use the PyTorch reference.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed; Phase E requires a CUDA Triton environment")
    if not packed.is_cuda:
        raise ValueError("Triton canonical INT4 unpack requires a CUDA tensor")
    if packed.dtype != torch.uint8:
        raise ValueError("packed canonical INT4 values must have dtype torch.uint8")
    if not packed.is_contiguous():
        raise ValueError("packed canonical INT4 values must be contiguous")
    available = packed.numel() * 2
    count = available if elements is None else elements
    if not isinstance(count, int) or count < 0 or count > available:
        raise ValueError(f"elements must be in [0, {available}], got {count!r}")
    output = torch.empty(count, device=packed.device, dtype=torch.int8)
    if count == 0:
        return output
    block = 256
    grid = (triton.cdiv(count, block),)
    _unpack_canonical_signed_int4_kernel[grid](packed, output, elements=count, BLOCK=block)
    return output


def dequantize_canonical_w4g64_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Dequantize canonical asymmetric W4G64 tensors to an FP32 CUDA weight.

    This primitive is intentionally limited to the frozen v6 layout. Matrix
    multiplication, bias addition, fusion, and performance tuning are outside
    its scope.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed; Phase E requires a CUDA Triton environment")
    if not isinstance(out_features, int) or out_features <= 0:
        raise ValueError("out_features must be a positive integer")
    if not isinstance(in_features, int) or in_features <= 0 or in_features % 64:
        raise ValueError("canonical asymmetric Triton dequantization requires in_features divisible by 64")
    for name, tensor, dtype in (
        ("qweight", qweight, torch.uint8),
        ("scales", scales, torch.float16),
        ("zeros", zeros, torch.uint8),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    groups_per_row = in_features // 64
    expected_qweight = out_features * in_features // 2
    expected_scales = out_features * groups_per_row
    expected_zeros = (expected_scales + 1) // 2
    if qweight.numel() != expected_qweight:
        raise ValueError(f"qweight must contain {expected_qweight} bytes, got {qweight.numel()}")
    if scales.numel() != expected_scales:
        raise ValueError(f"scales must contain {expected_scales} values, got {scales.numel()}")
    if zeros.numel() != expected_zeros:
        raise ValueError(f"zeros must contain {expected_zeros} bytes, got {zeros.numel()}")
    elements = out_features * in_features
    output = torch.empty((out_features, in_features), device=qweight.device, dtype=torch.float32)
    block = 256
    _dequantize_canonical_w4g64_kernel[(triton.cdiv(elements, block),)](
        qweight,
        scales,
        zeros,
        output,
        elements=elements,
        in_features=in_features,
        groups_per_row=groups_per_row,
        BLOCK=block,
    )
    return output


def linear_canonical_w4g64_triton(
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Run correctness-first canonical asymmetric W4G64 linear on CUDA Triton.

    Inputs are restricted to contiguous rank-2 FP16/BF16 activations and the
    frozen v6 packed layout. The defaults are the Phase E correctness baseline;
    documented alternatives are reserved for the fail-closed autotune runner.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed; Phase E requires a CUDA Triton environment")
    if activations.ndim != 2 or not activations.is_cuda or not activations.is_contiguous():
        raise ValueError("activations must be a contiguous rank-2 CUDA tensor")
    if activations.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("activations must have dtype torch.float16, torch.bfloat16, or torch.float32")
    rows, in_features = activations.shape
    if rows <= 0 or in_features <= 0 or in_features % 64:
        raise ValueError("activations require a positive W4G64-compatible feature dimension")
    for name, tensor, dtype in (
        ("qweight", qweight, torch.uint8),
        ("scales", scales, torch.float16),
        ("zeros", zeros, torch.uint8),
    ):
        if not tensor.is_cuda or not tensor.is_contiguous() or tensor.dtype != dtype:
            raise ValueError(f"{name} must be a contiguous CUDA tensor with dtype {dtype}")
    groups_per_row = in_features // 64
    if scales.numel() % groups_per_row:
        raise ValueError("scales do not encode an integral number of output rows")
    out_features = scales.numel() // groups_per_row
    if out_features <= 0 or qweight.numel() != out_features * in_features // 2:
        raise ValueError("qweight size does not match the canonical W4G64 layout")
    if zeros.numel() != (out_features * groups_per_row + 1) // 2:
        raise ValueError("zeros size does not match the canonical W4G64 layout")
    if bias is not None:
        if not bias.is_cuda or not bias.is_contiguous() or bias.ndim != 1 or bias.numel() != out_features:
            raise ValueError("bias must be a contiguous CUDA vector matching out_features")
        if bias.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("bias must have dtype torch.float16 or torch.bfloat16")
    if block_m not in (8, 16, 32) or block_n not in (32, 64, 128):
        raise ValueError("unsupported Phase E tile; use documented autotune candidates")
    if num_warps not in (2, 4, 8) or num_stages not in (2, 3, 4):
        raise ValueError("unsupported Phase E launch configuration")
    output = torch.empty((rows, out_features), device=activations.device, dtype=activations.dtype)
    block_k = 64
    _linear_canonical_w4g64_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    ](
        activations, qweight, scales, zeros, bias, output,
        rows=rows, out_features=out_features, in_features=in_features,
        groups_per_row=groups_per_row, HAS_BIAS=bias is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output
