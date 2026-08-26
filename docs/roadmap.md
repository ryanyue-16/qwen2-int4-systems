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
