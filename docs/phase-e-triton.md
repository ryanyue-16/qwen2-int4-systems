# Phase E Triton W4A16 kernel contract

## Scope

Phase E implements a correctness-first Triton operator for the frozen canonical
AWQ v6 W4G64 asymmetric contract. This phase does not include autotuning,
latency or throughput measurement, KV cache, batching, or a production claim.

The only supported packed layout is canonical signed INT4: K-even code in the
low nibble and K-odd code in the high nibble. Quantization is row-major
`[out_features, in_features]`; scales are `[out_features, in_features / 64]`;
and zero points are packed row-major signed INT4 values, one per output-group.

## Correctness reference

The oracle is `qwen_int4.backends.torch_reference_linear`, which first applies:

```text
dequantized_weight = (signed_code - signed_zero) * scale
output = activation @ dequantized_weight.T + bias
```

The Triton implementation must consume identical `qweight`, `scales`, and
`zeros` tensors. It may not reinterpret the legacy signed-add packing format,
silently substitute symmetric zero points, or accept group sizes other than 64.

## Required test matrix

Before a fused operator is accepted, every implemented primitive must be tested
against its PyTorch reference with deterministic random, zero, alternating-sign,
and bounded large-magnitude inputs. The first operator matrix uses these Qwen2
linear dimensions, written as `[M, N, K]`:

| Case | Shape | Purpose |
| --- | --- | --- |
| V projection | `[1, 256, 1536]` | Decode-sized GQA projection |
| Q projection | `[8, 1536, 1536]` | Square attention projection |
| MLP down projection | `[4, 1536, 8960]` | Wide-K MLP projection |

The kernel must reject invalid metadata, non-W4G64 group layouts, malformed
packed tensor sizes, and unsupported dtypes before execution.

## Numerical gates

All intermediate reference comparisons use FP32. The final output is compared
after conversion to the requested activation dtype:

| Output dtype | `rtol` | `atol` |
| --- | ---: | ---: |
| FP16 | `0.01` | `0.0625` |
| BF16 | `0.02` | `0.25` |

The tolerance applies to every required case. A failure blocks fusion,
autotuning, and any performance measurement. Passing proves numerical parity
only; it does not prove end-to-end model parity or performance.

## Implementation order

1. Validate the remote CUDA and Triton environment and record a non-performance
   environment report.
2. Implement canonical signed-INT4 unpacking with a PyTorch-oracle test.
3. Implement W4G64 per-group asymmetric dequantization with a PyTorch-oracle test.
4. Implement the fused linear operator and run the required operator matrix.
5. Review results before allowing any autotuning work.

The unpack primitive is intentionally isolated in `qwen_int4.triton_kernels`.
It accepts only contiguous CUDA `uint8` tensors and returns signed `int8`
values. Tests run only on CUDA hosts with Triton; non-CUDA development hosts
skip them rather than pretending to validate a kernel.

The first primitive passed its remote correctness suite on 2026-08-31. The
versioned evidence is `results/phase_e_triton_unpack_v1.json`; it records
canonical boundary patterns, random packed bytes, odd decoded element counts,
and invalid-contract rejection. It contains no performance measurement.

The W4G64 asymmetric dequantization primitive also passed its remote FP32
elementwise comparison suite on 2026-08-31. Its evidence is
`results/phase_e_triton_dequant_v1.json`; it covers odd packed zero-point
counts, multiple groups, and the Qwen2 V-projection weight shape. It does not
include matrix multiplication or a performance measurement.

The fixed-tile fused W4A16 linear operator passed its remote correctness suite
on 2026-08-31. `results/phase_e_triton_linear_v1.json` records comparison with
the canonical PyTorch backend for the Qwen2 V projection, Q projection, and
MLP down projection, plus zero, alternating-sign, and bounded large-magnitude
inputs. The result is numerical evidence only: the tile was not autotuned and
no benchmark was collected.

The final correctness gate compared the operator against the PyTorch oracle on
three representative tensors read from the SHA-verified frozen v6 checkpoint.
`results/phase_e_triton_v6_operator_parity_v1.json` records the exact layers,
input shapes, tolerances, and metrics. This is not an end-to-end model test and
does not provide a performance result.
