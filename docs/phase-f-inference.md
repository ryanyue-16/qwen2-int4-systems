# Phase F complete inference optimization

## Current correctness foundation

The model exposes an explicit per-layer KV cache through `past_key_values` and
`use_cache`. A prefill call returns logits and cache tensors shaped
`[batch, num_key_value_heads, cached_sequence, head_dim]`. Decode accepts the
cache plus only the new tokens, while its attention mask covers cached and new
positions.

The cache path is covered by deterministic equivalence tests for token-by-token
decode and batched multi-token decode. These tests establish output correctness
against a no-cache full forward only; they do not measure speed or memory.

`greedy_generate_cached` provides the first cache-aware generation API. It uses
one prefill followed by single-token cached decode and is regression-tested
against greedy generation that recomputes each complete prefix. Sampling and
beam duplication are not yet part of this API.

Batching currently permits only left-padded prompts: each attention-mask row
must be a contiguous valid suffix, and every row must contain at least one
token. This keeps the last prefill logit aligned with the final valid prompt
token while retaining a rectangular K/V cache. Right padding is rejected.

`CachedGenerationState.select_requests` now owns reorder and removal semantics:
indices must be unique and in range, and every per-layer K/V tensor is selected
with the same batch order. `close` releases the owned cache and makes later use
fail closed. Duplicate indices remain unsupported because they require an
explicit beam-cache ownership contract.

## Remaining Phase F order

1. Freeze a model-level benchmark protocol before collecting performance data.
2. Run reproducible prefill, decode, batching, memory, and end-to-end benchmarks.
3. Record the Phase F exit evidence and quality limitations.

No Phase F result may be called model latency, throughput, or production
performance until the final benchmark protocol records synchronization, warmup,
percentiles, memory, environment, and reproducible commands.

The dispatch accepts Triton only for canonical asymmetric W4G64 tensors on a
CUDA host. It rejects legacy packing and other group sizes rather than silently
falling back to a different implementation.

The isolated `QuantLinear` dispatch path passed its CUDA correctness suite on
2026-08-31, including the Phase E primitives and the dispatch-to-reference
comparison. Full-model Triton parity remains a separate Phase F integration
gate and is not implied by this layer-level result.

The first frozen-v6 full-model gate passed on 2026-08-31. BF16/FP16 canonical
W4G64 linears use the IEEE-accumulating Triton path, while FP32 SwiGLU
down-projection activations use an explicit PyTorch reference fallback. The
result therefore establishes a correctness-tested hybrid backend, not an
all-Triton model. Exact logits metrics and cached greedy tokens are recorded in
`results/phase_f_triton_model_parity_v1.json`.

The expanded frozen-v6 matrix passed all 11 Phase C deterministic prompt cases
with four cached greedy tokens per case. Generation tokens matched in 11/11
cases and the minimum prompt-logits cosine was `0.999700665473938`, above the
`0.9997` gate. The local summary is
`results/phase_f_correctness_matrix_summary_v1.json`; it records the SHA-256 of
the complete remote report. This remains correctness evidence, not a benchmark.
