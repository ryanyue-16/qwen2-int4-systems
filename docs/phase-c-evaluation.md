# Phase C unified quality evaluation

## Status

The Phase C protocol, datasets, model matrix, checksums, and runner are frozen.
The current repository status is `pending_model_runs`, not a completed quality
claim. The full reference-AWQ row requires the native Phase B checkpoint and an
NVIDIA Linux host.

The locally validated plan contains:

- the complete 4,358-row WikiText-2 raw test split from a pinned dataset commit;
- an independent project-authored Chinese evaluation corpus;
- 11 deterministic generation cases across English, Chinese, Python, arithmetic,
  logic, and long-context retrieval;
- identical tokenizer identity, token IDs, BOS/EOS behavior, scoring windows,
  stride, and greedy decoding parameters for every model;
- SHA-256 validation and strict calibration/evaluation path and split isolation;
- versioned, fail-closed JSON output with NLL, perplexity, perplexity ratio,
  top-1 agreement, target-token log-probability MAE, finite-value checks, runtime,
  peak-memory observations, and generation regressions.

## Frozen inputs

| Input | Use | SHA-256 |
| --- | --- | --- |
| `data/phase_c/wikitext2_test_full.txt` | English NLL/PPL | `90172ab2c5b6fce8f8a89a438733715ffef0989f220fac42e4fcfdd9757fe1f6` |
| `data/phase_c/chinese_eval_v1.txt` | Chinese NLL/PPL | `590a5dacceafbda5ca01f509c07999ad041d81c51fe64f17edce9075fa238afb` |
| `data/phase_c/generation_prompts_v1.json` | Generation regression | `1494167d293db82128a4769b639eb191c7c92e6cca36db28b23616f758e2e164` |

WikiText-2 is fixed at the
[Salesforce dataset commit](https://huggingface.co/datasets/Salesforce/wikitext/commit/b08601e04326c79dfdd32d625aee71d232d685c3).
The test parquet SHA-256 is recorded separately in
`data/phase_c/wikitext2_test_full.metadata.json`. Corpus preparation removes
trailing whitespace from each source row and records that normalization. AWQ
reference calibration uses the train split from the same commit. The test split
is never permitted as a calibration input.

## Frozen protocol

- tokenizer and model revision: `8a16abf2848eda07cc5253dec660bf1ce007ad7a`;
- no automatically inserted special tokens for corpus scoring or generation;
- sliding window: 512 tokens;
- stride: 256 tokens;
- NLL accumulation: float64;
- decoding: greedy, no sampling, 32 new tokens;
- reports: new versioned path required; overwrite is not supported.

The 512-token scoring window is a measurement choice shared by all rows. It is
not a model context-length claim. The generation heuristics detect empty output,
replacement characters, low token diversity, and positional divergence from
BF16. Human review is still required for gibberish, repetition, and obvious
logic collapse because a token-level heuristic is not sufficient.

## Model matrix

| Model | Artifact status | Formal Phase C result |
| --- | --- | --- |
| Official BF16 | ready | pending |
| W4G64 RTN v4 | ready | pending |
| W4G128 AWQ reference v1 | `pending_gpu_validation` | pending |
| W4G128 canonical self-AWQ v1 | `awaiting_phase_d` | pending |

No blank cell is treated as a pass. The final report is complete only when all
four rows use the same frozen inputs and protocol.

## Commands

Validate inputs, checksums, isolation, and artifact availability without loading
a model:

```bash
PYTHONPATH=src python scripts/validate_phase_c.py
PYTHONPATH=src python scripts/run_phase_c_evaluation.py --validate-only
```

Run a short development check. Reports produced with `--max-tokens` are marked
non-formal and cannot satisfy Phase C:

```bash
PYTHONPATH=src python scripts/run_phase_c_evaluation.py \
  --models bf16,w4g64_rtn_v4 \
  --device cuda \
  --max-tokens 2048 \
  --output-json results/phase_c_development_2k_v1.json
```

Run the full protocol after the Phase B reference artifact is GPU-validated and
the Phase D artifact exists:

```bash
PYTHONPATH=src python scripts/run_phase_c_evaluation.py \
  --models bf16,w4g64_rtn_v4,w4g128_awq_reference_v1,w4g128_awq_canonical_v1 \
  --device cuda \
  --output-json results/phase_c_evaluation_v1_gpu_run1.json
```

The full test corpus is intentionally not run as a background CPU benchmark on
the current Mac. Its result must be captured together with the AWQ rows on the
same NVIDIA evaluation host to keep timing and peak-memory context comparable.
