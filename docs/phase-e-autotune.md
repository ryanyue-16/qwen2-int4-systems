# Phase E Triton autotune protocol

## Scope

This protocol selects fixed launch parameters for the validated canonical W4G64
asymmetric Triton operator. It is an operator-level measurement only. It does
not measure a full model, prefill, decode, batching, KV cache, or production
service performance.

## Fail-closed procedure

For each profile and candidate, the runner first compares one operator output
against `torch_reference_linear` using the profile tolerance. A candidate that
fails numerical parity is excluded and is never timed. Passing candidates are
warmed up, timed with CUDA events, and ranked by median kernel milliseconds.

The candidate set, warmup count, measurement count, profiles, tolerances, and
selection rule are frozen in `configs/phase-e-triton-autotune-v1.json`.
The runner refuses an existing output path and verifies the frozen v6 checkpoint
SHA-256 before reading any tensor.

## Interpretation

The selected launch settings are valid only for the listed GPU environment,
operator profiles, and frozen artifact. Median kernel time is not model latency
or throughput. Any Phase F claim requires its own prefill/decode, batching, and
end-to-end measurement protocol.

## Completed v1 result

The v1 run used the frozen v6 checkpoint on the recorded RTX 4090 environment.
All 15 candidate/profile parity checks passed. The selected launch settings for
each listed profile were `BLOCK_M=8`, `BLOCK_N=32`, `num_warps=2`, and
`num_stages=3`. The versioned result is
`results/phase_e_triton_autotune_v1.json`. Its median CUDA-event kernel times
are operator measurements only and must not be presented as model latency or
throughput.
