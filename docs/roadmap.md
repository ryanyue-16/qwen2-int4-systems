# Project roadmap

## Phase A — Frozen RTN baseline

- [x] Preserve W4G64 RTN v4 and verify its SHA-256.
- [x] Pin the Qwen2-1.5B model and tokenizer revisions.
- [x] Capture the artifact, data, report, software, and hardware manifest.
- [x] Add an exact local dependency lock and unified fail-closed reproduce command.
- [x] Re-run all 15 existing tests in the captured `qwen_vl` environment.
- [x] Prevent silent replacement of versioned evaluation outputs.

Exit evidence is documented in `docs/reproducibility.md`. The checkpoint remains
the comparison baseline for phases B–D. No Triton, KV-cache, CUDA-performance,
or real-LPU claim is part of phase A.

## Phase B — Mature AWQ reference

- [x] Select an actively maintained official AWQ implementation with Qwen2 mappings.
- [x] Freeze the W4G128 asymmetric recipe, calibration source, and native format.
- [x] Add an isolated, version-pinned NVIDIA Linux environment.
- [x] Add fail-closed quantization and quality-evaluation entry points.
- [ ] Produce and validate the native checkpoint on an NVIDIA Linux host.

The selected oracle is LLM Compressor 0.13.0 with native compressed-tensors
export. The local status is `pending_gpu_validation`; no reference quality or
performance number has been fabricated. See `docs/awq-reference.md`.

## Phase C — Unified quality evaluation

- [x] Freeze the complete WikiText-2 test input and an independent Chinese corpus.
- [x] Expand deterministic English, Chinese, code, math/logic, and long-context cases.
- [x] Enforce shared tokenization, scoring, decoding, checksum, and isolation rules.
- [x] Add a unified fail-closed runner for BF16, RTN, reference AWQ, and self-AWQ.
- [ ] Run all four model rows on one NVIDIA evaluation host and apply the gates.

The protocol is ready, but Phase C is not numerically complete while the Phase B
reference and Phase D checkpoints are absent. See `docs/phase-c-evaluation.md`.

## Phase D — Canonical self-AWQ

- [x] Add memory-bounded sequential calibration primitives.
- [x] Add Qwen2 Q/K/V, GQA V/O, and SwiGLU mappings.
- [x] Add recoverable per-channel scale search and equivalent scale migration.
- [x] Add activation-weighted group-wise clipping search.
- [x] Add asymmetric canonical W4G128 packing, zero points, and export metadata.
- [x] Add CPU unit gates for transformations, search safety, and export loading.
- [ ] Run end-to-end Qwen2 calibration and validate against the NVIDIA reference.
- [ ] Pass the complete Phase C quality matrix and freeze the final checkpoint.

The current status is `awaiting_gpu_reference`; CPU implementation progress is
not a reference-quality claim. See `docs/phase-d-canonical-awq.md`.

## Phase 1 — Recovered reference repository (current)

- [x] Inspect the 730-tensor checkpoint contract.
- [x] Make the legacy signed-add packing rule explicit.
- [x] Add canonical and legacy pack/unpack round-trip tests.
- [x] Add strict, fail-closed checkpoint loading.
- [x] Replace environment-variable dispatch with explicit backends.
- [x] Run full-model CPU smoke tests for torch and LPU functional paths.
- [x] Compare dequantized sampled weights with the original BF16 model (failed: cosine approximately zero).
- [x] Add intermediate activation capture and first-divergence reports.

Exit criterion: packing format is independently confirmed and Qwen2 architecture parity is measured layer by layer.

The old checkpoint remains blocked. The selected replacement is the canonical
W4G64 RTN v4 checkpoint. Backend/architecture parity reaches `0.99987543`
final-logits cosine. Two held-out smoke corpora show lower NLL than W4G128;
activation clipping was rejected because its gain did not reproduce. See
`docs/quality-analysis.md`.

## Phase 2 — Triton learning path

- [ ] FP16/BF16 tiled GEMM.
- [ ] Standalone packed INT4 unpack kernel.
- [ ] Group-wise dequantization kernel.
- [ ] Fused load → unpack → scale → GEMM W4A16 kernel.
- [ ] Autotune `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, and `num_warps` only after correctness passes.

Exit criterion: Triton matches the PyTorch reference under documented tolerances for representative Qwen2 linear shapes.

## Phase 3 — Inference and evaluation

- [ ] KV cache and separate prefill/decode paths.
- [ ] Kernel, operator, layer, prefill, and decode benchmarks.
- [ ] Warmup, synchronization, percentiles, environment manifest, and peak memory reporting.
- [x] Initial BF16 versus W4A16 held-out NLL/perplexity evaluation.
- [x] 2K-token WikiText-2 sliding-window quality gate.
- [x] Deterministic Chinese/English/code/arithmetic generation smoke suite.
- [ ] Broader task-specific evaluation before any production claim.
- [ ] Compare with one production W4 backend as context, without using its kernel as this project's implementation.

Exit criterion: the repository can answer “does it work, is it faster, and how much quality is lost?” with reproducible evidence.
