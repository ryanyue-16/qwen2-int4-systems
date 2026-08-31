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
- [x] Produce and validate the native checkpoint on an NVIDIA Linux host.

The selected oracle is LLM Compressor 0.13.0 with native compressed-tensors
export. It completed NVIDIA execution, but its final Phase C quality ratios did
not meet the `<= 1.05` selection gate. See `docs/awq-reference.md`.

## Phase C — Unified quality evaluation

- [x] Freeze the complete WikiText-2 test input and an independent Chinese corpus.
- [x] Expand deterministic English, Chinese, code, math/logic, and long-context cases.
- [x] Enforce shared tokenization, scoring, decoding, checksum, and isolation rules.
- [x] Add a unified fail-closed runner for BF16, RTN, reference AWQ, and self-AWQ.
- [x] Run the user-approved narrowed three-model v3 matrix and apply the gates.

The original four-row v1 matrix remains historical and incomplete because the
RTN baseline was intentionally removed from remote execution. V3 is complete
for its documented narrowed scope. See `docs/phase-c-evaluation.md`.

## Phase D — Canonical self-AWQ

- [x] Add memory-bounded sequential calibration primitives.
- [x] Add Qwen2 Q/K/V, GQA V/O, and SwiGLU mappings.
- [x] Add recoverable per-channel scale search and equivalent scale migration.
- [x] Add activation-weighted group-wise clipping search.
- [x] Add asymmetric canonical W4G128 packing, zero points, and export metadata.
- [x] Add CPU unit gates for transformations, search safety, and export loading.
- [x] Run end-to-end Qwen2 calibration and validate v6 against the HF reference.
- [x] Pass the narrowed Phase C v3 quality gates and freeze the v6 checkpoint contract.

The selected v6 manifest is the sole input contract for Phase E. Phase D does
not establish a kernel or performance result. See `docs/phase-d-canonical-awq.md`.

## Phase E — Triton W4A16 kernel

- [x] Freeze the W4G64 asymmetric kernel correctness specification and tests.
- [x] Validate the remote Triton/CUDA development environment without benchmarks.
- [x] Implement and validate standalone canonical packed signed-INT4 unpacking.
- [x] Implement and validate per-group asymmetric W4G64 dequantization.
- [x] Implement the correctness-first fused W4A16 linear operator.
- [x] Prove numerical parity with the canonical PyTorch backend at representative Qwen2 shapes.
- [x] Verify operator parity on representative tensors from the frozen v6 checkpoint.
- [x] Run fail-closed operator autotuning only after all numerical correctness gates pass.

Exit criterion: the Triton operator meets the documented numerical contract for
canonical v6 data. This does not establish latency, throughput, or production
performance. See `docs/phase-e-triton.md` and `docs/phase-e-autotune.md`.

## Phase F — Complete inference optimization

- [x] Add correctness-tested KV cache and separate prefill/decode execution paths.
- [x] Add left-padded batching and correctness-tested hybrid Triton model integration.
- [x] Add request reorder/removal, cache release, and an 11-case model correctness matrix.
- [x] Run reproducible operator, layer, prefill, decode, batching, and model benchmarks.
- [x] Capture warmup, synchronization, percentile, memory, and environment evidence.

Exit criterion: reproducible evidence can answer whether the complete inference
stack works, how it performs, and what quality trade-offs remain.

Phase F correctness and benchmark boundaries are documented in
`docs/phase-f-inference.md`; final evidence is summarized in
`docs/phase-f-report.md`.

## Phase G — LPU functional emulation and analytical mapping

- [x] Define the PyTorch, Triton GPU, LPU emulator, analytical mapping, and
  real-LPU validation boundaries.
- [x] Map frozen canonical W4G64 tensors to an analytical A/B/C-buffer dataflow.
- [x] Make the functional emulator fail closed on incompatible contracts and
  malformed metadata.
- [x] Add deterministic CPU parity, edge-case, rejection, dispatch, and
  multi-layer smoke tests.
- [x] Document numerical tolerances, non-goals, and future real-LPU requirements.

Phase G is complete only as an analytical/emulation extension. It does not add
artifact governance, nightly infrastructure, dashboards, a real compiler or
runtime integration, real-LPU correctness, or real-LPU performance. See
`docs/phase-g-lpu-emulation.md`.
