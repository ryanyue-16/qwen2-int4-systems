# Qwen2 INT4 Systems

A correctness-first project for reproducible Qwen2-1.5B W4A16 quantization and
inference research. The repository currently provides:

- an explicitly versioned packed INT4 checkpoint contract;
- a PyTorch numerical reference backend;
- an LPU-style K-streaming functional backend;
- explicit backend selection without import-time environment variables;
- Qwen2 RMSNorm, RoPE, GQA, causal attention, SwiGLU, and CausalLM components;
- checkpoint inspection, packing validation, quality evaluation, and tests;
- a frozen W4G64 RTN baseline for fair AWQ comparisons.
- a pinned mature AWQ reference recipe awaiting NVIDIA GPU validation.

Triton kernels, KV caching, and performance claims are intentionally out of
scope until the AWQ checkpoint contract and quality gate are frozen. The current
LPU backend is a functional model, not a real hardware runtime.

## Current baseline

The recommended checkpoint is:

```text
artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors
```

Its frozen SHA-256 is:

```text
6181db4f9542b7db3a5ea4458b5c11b50ad2da51cfd20c33188d58efd43b147c
```

Verified properties:

- symmetric RTN W4G64 with code range `[-7, 7]`;
- row-major weights with shape `[out_features, in_features]`;
- K-even in the low nibble and K-odd in the high nibble;
- scales shaped `[out_features, in_features / group_size]`;
- zero points are all zero;
- 28 decoder layers, 196 quantized linear layers, and 588 quantized tensors;
- worst direct weight cosine: `0.98668426`;
- architecture/backend final-logits cosine: `0.99987543`;
- 15 tests passing in the captured environment.

The checkpoint contains only trusted `qweight`, `scales`, and `zeros` tensors.
Embedding, normalization, bias, and tied LM-head parameters are loaded from the
pinned official Hugging Face model.

## Quality snapshot

The frozen WikiText-2 2K-token result is:

| Metric | BF16 | W4G64 RTN |
| --- | ---: | ---: |
| NLL | 2.468979 | 2.560360 |
| Perplexity | 11.8104 | 12.9405 |
| PPL ratio | 1.000000 | 1.095686 |

Additional measurements:

- top-1 agreement: `0.8608`;
- target-token log-probability MAE: `0.285803`;
- deterministic generation mean position agreement: `0.8125`;
- Python and arithmetic generation cases: `100%` position agreement.

These numbers establish a reproducible comparison baseline. They are not a
production-readiness claim. See [Quality analysis](docs/quality-analysis.md) and
[Validation report](docs/validation-report.md) for details.

## Pinned upstream identity

All BF16 model and tokenizer loaders default to the same Hugging Face snapshot:

```text
model: Qwen/Qwen2-1.5B
model revision: 8a16abf2848eda07cc5253dec660bf1ce007ad7a
tokenizer revision: 8a16abf2848eda07cc5253dec660bf1ce007ad7a
```

The single source of truth is `src/qwen_int4/provenance.py`. CLI users may
override the revisions with `--hf-revision` and `--tokenizer-revision`; new
quantization metadata and evaluation reports record the effective revisions.
The existing RTN v4 artifact is never migrated or silently replaced.

The revision is a retrospective pin to the locally validated snapshot because
the original v4 checkpoint did not embed a model commit. This limitation is
recorded explicitly in the artifact manifest.

## Repository layout

```text
.
├── configs/qwen2-1.5b/config.json
├── data/                              # Calibration and evaluation inputs
├── docs/
│   ├── architecture/                  # Public high-level LPU mapping notes
│   ├── quality-analysis.md
│   ├── reproducibility.md
│   ├── roadmap.md
│   └── validation-report.md
├── scripts/
│   ├── capture_environment.py
│   ├── evaluate_generation.py
│   ├── evaluate_quality.py
│   ├── fetch_wikitext2.py
│   ├── inspect_checkpoint.py
│   ├── quantize_model.py
│   ├── reproduce_rtn_v4.py
│   └── validate_against_hf.py
├── src/qwen_int4/
│   ├── backends.py
│   ├── checkpoint.py
│   ├── cli.py
│   ├── linear.py
│   ├── model.py
│   ├── provenance.py
│   ├── quantization.py
│   └── validation.py
├── tests/
├── pyproject.toml
├── requirements-rtn-v4.txt
└── requirements.txt
```

Chinese text under `data/` is intentional evaluation content for multilingual
quality coverage. Public documentation, code comments, and contributor-facing
text are written in English.

## Installation

Create an isolated environment:

```bash
cd /path/to/qwen2-int4-systems
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The exact macOS arm64 baseline dependencies are recorded in
`requirements-rtn-v4.txt`. On NVIDIA systems, install the appropriate official
CUDA-enabled PyTorch build before installing this project.

## Reproduce the frozen baseline

On the captured local conda environment:

```bash
PYTHONPATH=src conda run --no-capture-output -n qwen_vl \
  python scripts/reproduce_rtn_v4.py
```

This fail-closed command verifies:

- the manifest schema and every registered checksum;
- model and tokenizer revisions;
- exact Python package versions;
- calibration/evaluation path separation;
- checkpoint metadata, tensor counts, packing, and zero points;
- all 15 tests.

It does not overwrite artifacts or results. Use `--full` to create new,
timestamped parity, quality, and generation reports:

```bash
PYTHONPATH=src conda run --no-capture-output -n qwen_vl \
  python scripts/reproduce_rtn_v4.py --full --device cpu
```

See [Reproducibility contract](docs/reproducibility.md) for environment and
provenance details.

## Mature AWQ reference

Phase B selects vLLM Project LLM Compressor 0.13.0 as the external W4G128 AWQ
oracle. The repository includes a frozen asymmetric recipe, a separate NVIDIA
Linux environment, and fail-closed quantization and evaluation entry points.
The native compressed-tensors format is preserved and is not assumed to match
this project's canonical packing. The current Mac cannot execute the CUDA run,
so the artifact is explicitly marked `pending_gpu_validation`.

See [Mature AWQ reference](docs/awq-reference.md) for the decision record and
NVIDIA commands.

## Unified quality gate

Phase C freezes the complete WikiText-2 test split, an independent Chinese
corpus, and deterministic multilingual/code/reasoning/long-context generation
cases. A single runner applies identical tokenization, scoring, and decoding to
BF16, RTN, external AWQ, and future canonical self-AWQ rows. The protocol is
ready, while the four-row NVIDIA result remains pending.

See [Phase C unified quality evaluation](docs/phase-c-evaluation.md).

## Canonical self-AWQ development

Phase D has started with CPU-testable calibration, Qwen2-aware scale migration,
recoverable scale and clipping searches, and a versioned asymmetric W4G128
canonical export contract. The implementation is marked
`awaiting_gpu_reference`; it is not yet a completed AWQ checkpoint or quality
claim.

See [Phase D canonical AWQ](docs/phase-d-canonical-awq.md).

## Inspect the checkpoint

```bash
PYTHONPATH=src python scripts/inspect_checkpoint.py \
  artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors
```

Expected output includes 28 layers, 196 quantized linears, 588 tensors, no
nonzero zero-point tensors, `format_version=qwen-int4-v1`, and canonical packing.

## Run the tests

```bash
PYTHONPATH=src python -m pytest
```

The suite covers canonical and legacy pack/unpack behavior, raw byte layout,
group-scale indexing, fail-closed zero-point handling, quality metrics, and
elementwise agreement between the PyTorch and LPU-style functional backends.

## Run an inference smoke test

Using token IDs avoids loading a tokenizer:

```bash
PYTHONPATH=src python -m qwen_int4.cli \
  --checkpoint artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors \
  --input-ids 9707,11 \
  --backend torch \
  --device cpu
```

Natural-language prompt example:

```bash
PYTHONPATH=src python -m qwen_int4.cli \
  --checkpoint artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors \
  --prompt "Briefly explain weight-only quantized inference." \
  --max-new-tokens 8 \
  --backend torch \
  --device cpu
```

Generation currently recomputes the full prefix for every new token because KV
caching is not implemented. This path is for correctness checks, not benchmarks.

## Checkpoint format

Canonical packing stores two signed INT4 values in each byte:

```text
byte = (q[K-odd] & 0x0f) << 4 | (q[K-even] & 0x0f)
```

The historical unverified checkpoint may have used signed addition:

```python
packed = (high << 4) + low
```

A negative signed low nibble borrows from the high nibble. The
`legacy_signed_add` decoder models this behavior for forensic analysis, but
histogram evidence alone cannot establish agreement with official BF16 weights.
All backends therefore reuse `quantization.py` as the only unpacking source.

## Re-quantize from official BF16 weights

The quantizer refuses to overwrite an existing output unless explicitly asked:

```bash
PYTHONPATH=src python scripts/quantize_model.py \
  --hf-model Qwen/Qwen2-1.5B \
  --hf-revision 8a16abf2848eda07cc5253dec660bf1ce007ad7a \
  --output artifacts/qwen2-1.5b-w4g64-rtn-new/model.safetensors \
  --group-size 64 \
  --local-files-only
```

Use a new versioned directory. Never target the frozen v4 path during routine
experimentation.

## Scope and non-claims

- No Triton kernel is implemented in the current phase.
- No KV cache or optimized prefill/decode split is implemented.
- macOS CPU timings must not be presented as NVIDIA or LPU performance.
- The LPU-style Python backend is not bit-accurate, cycle-accurate, or a real
  hardware runtime.
- External AWQ formats must not be assumed to match the canonical packing used
  by this repository.

The staged implementation plan is maintained in [Roadmap](docs/roadmap.md).
