# Phase G LPU functional emulation

## Scope and terminology

Phase G is a correctness-first analytical extension of the frozen canonical
W4G64 asymmetric AWQ contract. It defines the following terms:

- **PyTorch reference:** dequantize the complete packed weight to FP32, execute
  one FP32 matrix multiplication, add optional bias, and cast to the activation
  dtype.
- **Triton GPU backend:** execute the validated hybrid GPU path described in
  the Phase E and Phase F documents. The complete model is not all-Triton.
- **LPU functional emulator:** execute canonical W4G64 linear layers with
  K-group streaming and FP32 accumulation using PyTorch tensor operations.
- **Analytical dataflow mapping:** describe how canonical tensors could flow
  through abstract A, B, and C buffers without assigning physical capacities,
  instruction encodings, timing, or a hardware schedule.
- **Real LPU validation:** compile and execute with a specified LPU toolchain and
  runtime, then validate numerical behavior and measurements on identified
  hardware under a separately reviewed protocol.

The current `backend="lpu"` path is only an LPU-style functional emulator. It
is not cycle-accurate, bit-accurate, a compiler/runtime integration, real LPU
execution, or real LPU performance evidence.

## Frozen W4G64 input contract

The emulator accepts only the canonical asymmetric W4G64 representation:

- logical weights have row-major shape `[out_features, in_features]`;
- each byte stores K-even signed INT4 in the low nibble and K-odd signed INT4
  in the high nibble;
- packed zero points use the same signed INT4 encoding and row-major
  `[out_features, in_features / 64]` logical layout;
- FP16 scales have shape `[out_features, in_features / 64]`;
- dequantization is `(signed_code - signed_zero) * scale` for each K group;
- group size is exactly 64;
- AWQ-transformed normalization weights, linear weights, and applicable biases
  are checkpoint parameters consumed before or by the linear operation. The
  emulator does not recompute or select AWQ transformations.

Legacy packing, another group size, malformed tensor sizes or dtypes, invalid
scales, inconsistent devices, and mismatched activation or bias shapes are
rejected before execution. Non-contiguous activations are accepted: reshape
may materialize a contiguous logical view without changing the calculation.

## Analytical A/B/C dataflow

For an activation matrix `X[M, K]` and logical weight `W[N, K]`, define
symbolic positive tile dimensions `M_tile` and `N_tile`. These names are useful
for reasoning about partitioning only. They are not configured hardware
capacities and imply no SRAM size, bandwidth, cycle count, or instruction
format.

For every symbolic output tile and each 64-element K group:

1. The A buffer supplies an activation slice `A[:, k:k+64]` in logical FP16,
   BF16, or FP32 form.
2. The B path reads the packed weight codes, the packed asymmetric zero point,
   and the FP16 scale for that output channel and group.
3. The B slice is functionally dequantized to FP32 with
   `(code - zero) * scale`.
4. `A @ B.T` contributes to an FP32 C accumulator.
5. After all K groups, optional bias is added in FP32.
6. The output is cast to the input activation dtype and restored to the input
   leading dimensions.

The Python implementation currently traverses K groups while processing all
logical M rows and N outputs at once. Splitting M and N by the symbolic tile
dimensions is mathematically compatible with this mapping, but no physical
tiling or scheduling claim is made.

## Numerical validation

Deterministic CPU tests compare `lpu_functional_linear` with
`torch_reference_linear` for:

- canonical asymmetric W4G64 tensors, including AWQ-transformed-compatible
  weights and bias;
- FP16 and BF16 outputs, with and without bias, across multiple row counts;
- zero, large-magnitude, alternating-sign, and non-contiguous activations;
- signed INT4 zero points at `-8` and `7`;
- `QuantLinear` dispatch and a two-layer functional smoke path;
- deterministic rejection of wrong group size, wrong packing, inconsistent
  shapes, storage dtypes, and non-finite or non-positive scales.

The normal comparison tolerance is `rtol=2e-3, atol=2e-3`; large-magnitude and
non-contiguous edge cases use `atol=2e-2`. The two-layer smoke test uses
`rtol=3e-3, atol=3e-3`. The tolerance covers a different FP32 reduction order
between one full PyTorch matrix multiplication and sequential group partial
sums, followed by FP16 or BF16 output conversion. It is not a hardware error
model.

## Non-goals and limitations

Phase G does not provide or establish:

- private LPU architecture details, instruction formats, or compiler behavior;
- physical buffer capacity, bandwidth, memory capacity, scheduling, or cycles;
- hardware rounding, saturation, data movement, synchronization, or faults;
- real LPU model correctness, latency, throughput, power, or utilization;
- production deployment, artifact governance, dashboards, or nightly jobs.

The emulator uses PyTorch FP32 matrix multiplication and accumulation. Passing
its tests proves only the specified functional numerical behavior.

## Requirements for future real-LPU validation

A separate real-LPU phase would need an identified public compiler/runtime and
hardware target, a canonical tensor lowering and instruction mapping, explicit
rounding and accumulation semantics, device-side correctness evidence, and a
reviewed measurement protocol. Only measurements collected through that path
could support real-LPU latency, throughput, bandwidth, capacity, or compiler
compatibility statements.
