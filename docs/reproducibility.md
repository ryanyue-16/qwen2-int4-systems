# RTN v4 reproducibility contract

This document freezes the comparison baseline used by phases B–D. It does not
claim CUDA, Triton, or LPU performance.

## Frozen identities

- Model and tokenizer: `Qwen/Qwen2-1.5B`
- Model and tokenizer revision: `8a16abf2848eda07cc5253dec660bf1ce007ad7a`
- Checkpoint: `artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors`
- Checkpoint SHA-256: `6181db4f9542b7db3a5ea4458b5c11b50ad2da51cfd20c33188d58efd43b147c`
- Manifest schema: `qwen-int4-artifact-manifest-v1`

The upstream revision was added after the v4 file was produced. It is therefore
recorded as a retrospective pin to the local snapshot used for validation, not
as provenance embedded by the original quantization run. The baseline manifest
was captured before this directory was initialized as a Git repository, so the
existing checkpoint cannot be attributed to a source commit retroactively.

## Captured environment

The exact local macOS arm64 package set is in `requirements-rtn-v4.txt`. The
authoritative machine-readable record is
`artifacts/qwen2-1.5b-w4g64-rtn-v4/manifest.json`. It includes OS, CPU,
CUDA/MPS availability, Python and package versions, random seeds, checkpoint
metadata, and SHA-256 records for calibration/evaluation inputs and reports.

Install the captured Python dependencies in an isolated environment:

```bash
python -m pip install -r requirements-rtn-v4.txt
python -m pip install -e . --no-deps
```

The captured PyTorch wheel is platform-specific. On NVIDIA systems, install the
appropriate official CUDA PyTorch build first, install the remaining pinned
packages, and capture a new environment manifest. Do not describe that distinct
environment as byte-for-byte identical to the macOS baseline.

## Unified integrity command

On this machine:

```bash
PYTHONPATH=src conda run --no-capture-output -n qwen_vl \
  python scripts/reproduce_rtn_v4.py
```

The command fails closed on a manifest schema mismatch, changed/missing file,
SHA-256 mismatch, model/tokenizer revision mismatch, package version mismatch,
non-isolated calibration/evaluation paths, checkpoint inspection failure, or
test failure. A successful run verifies 15 tests and does not overwrite any
artifact or result.

## Full quality reproduction

The expensive local quality gates are opt-in:

```bash
PYTHONPATH=src conda run --no-capture-output -n qwen_vl \
  python scripts/reproduce_rtn_v4.py --full --device cpu
```

This creates three new reports with one UTC run identifier:

- HF architecture/backend parity;
- WikiText-2 2K-token quality comparison;
- deterministic generation regression.

It never targets the frozen result filenames. Individual evaluation scripts
also reject an existing output path unless `--overwrite` is explicitly passed.
CPU elapsed time is provenance/debug information only and must not be reported
as NVIDIA or LPU performance.

## Refreshing the manifest

Only refresh the manifest when intentionally capturing a new environment or
adding provenance. The script checks the frozen checkpoint SHA before writing
and refuses to replace an existing manifest without `--overwrite`:

```bash
PYTHONPATH=src python scripts/capture_environment.py \
  --input reserved_calibration=data/calibration_prompts.txt \
  --input held_out_evaluation=data/evaluation_corpus.txt \
  --input held_out_confirmation=data/confirmation_corpus.txt \
  --input wikitext2_test_slice=data/wikitext2_test_100rows.txt \
  --input generation_prompts=data/generation_prompts.json \
  --related local_model_config=configs/qwen2-1.5b/config.json \
  --related wikitext2_metadata=data/wikitext2_test_100rows.metadata.json \
  --related quantization_report=artifacts/qwen2-1.5b-w4g64-rtn-v4/quantization-report.json \
  --related hf_validation=results/hf_validation_w4g64_rtn_v4.json \
  --related wikitext2_quality=results/quality_w4g64_rtn_v4_wikitext2_2k.json \
  --related generation_regression=results/generation_w4g64_rtn_v4.json \
  --overwrite
```

Calibration and evaluation currently use distinct files and distinct hashes.
The broader semantic leakage audit belongs to the expanded phase-C dataset and
is not represented as completed by the phase-A manifest.
