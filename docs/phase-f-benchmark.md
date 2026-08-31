# Phase F model benchmark protocol

## Scope

This protocol measures the frozen canonical v6 model on the recorded RTX 4090
environment after all Phase F correctness gates pass. It compares the canonical
PyTorch reference backend with the correctness-tested hybrid Triton backend.
The hybrid backend uses Triton for FP16/BF16 canonical W4G64 linears and an
explicit PyTorch fallback for FP32 SwiGLU down projections.

## Measurement rules

The frozen profiles and iteration counts are defined in
`configs/phase-f-model-benchmark-v2.json`. Inputs are deterministic synthetic
token IDs, so tokenization and data loading are excluded. Model loading and
checkpoint loading are also excluded.

CUDA events measure prefill, one-token cached decode at the profile context
length, and complete cached greedy generation. Every series is warmed up and
synchronized. Reports include p50, p95, p99, minimum, maximum, and mean rather
than a single best timing. Peak allocated GPU memory is reset and recorded for
one complete cached generation after warmup.

## Interpretation limits

These results apply only to the listed checkpoint, software environment, GPU,
batch sizes, and sequence lengths. They are not production service latency,
multi-GPU throughput, scheduler performance, or real-LPU evidence. The PyTorch
backend is a correctness reference that dequantizes full weights on every call;
it is not a production baseline.
