# INT4 quality analysis and selected fix

## Conclusion

The current recommended checkpoint is
`artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors`. Reducing the RTN group
size from 128 to 64 improved held-out language-model NLL on both local corpora,
while retaining the same explicit canonical packing contract.

The activation-weighted clipping experiments are retained as negative results.
They are not the default because their gains did not reproduce on the confirmation
corpus.

The final project quality gate also passes on 2,048 tokens from the WikiText-2
raw test split: BF16 PPL is `11.8104`, W4G64 PPL is `12.9405`, and the ratio is
`1.0957` (+9.57%). This is below the gate fixed before evaluation (`+15%`). The
W4G64 v4 artifact is therefore the selected final quality version for the current
project phase.

## Standard-corpus quality gate

The source corpus is the first 100 rows of `Salesforce/wikitext`, configuration
`wikitext-2-raw-v1`, test split. The saved corpus contains 5,496 Qwen tokens;
the gate uses the first 2,048 with a 256-token context window and 128-token
stride. Every one of the 2,047 target positions is scored exactly once.

| Metric | BF16 | W4G64 v4 |
|---|---:|---:|
| NLL | 2.468979 | 2.560360 |
| Perplexity | 11.8104 | 12.9405 |
| Perplexity ratio | 1.0000 | 1.0957 |

Additional comparisons are `0.091381` excess NLL, `0.8608` top-1 agreement,
and `0.285803` mean absolute target-token log-prob difference. The corpus file
SHA-256 is `c2ee393ec598528c5a206b2f7760b9cb89d656f17bb66f5f8052fb6a1d022c09`.

Four deterministic eight-token generation cases give 81.25% mean positional
agreement. Python and arithmetic cases match BF16 exactly. Chinese and English
cases diverge after a shared prefix but remain locally coherent; no collapse or
invalid repetition was observed. These cases are regression smoke tests, not an
instruction-following benchmark, because the reference is the base model rather
than an instruct model.

## Reproducible measurements

All numbers below use the official `Qwen/Qwen2-1.5B` BF16 model as reference.
The two corpora are disjoint from the calibration prompts.

| Checkpoint | Corpus | Tokens | INT4 PPL | Excess NLL vs BF16 | Top-1 agreement | Logits cosine |
|---|---:|---:|---:|---:|---:|---:|
| W4G128 RTN v1 | A | 98 | 73.6871 | 0.188377 | 0.8351 | 0.983529 |
| W4G128 aggressive clip v2 | A | 98 | 79.7541 | 0.267498 | 0.8041 | 0.985687 |
| W4G128 conservative clip v3 | A | 98 | 70.8358 | 0.148915 | 0.8144 | 0.983035 |
| **W4G64 RTN v4** | **A** | **98** | **70.2154** | **0.140118** | **0.7320** | **0.985946** |
| W4G128 RTN v1 | B | 97 | 98.6573 | 0.167102 | 0.7083 | 0.981052 |
| W4G128 conservative clip v3 | B | 97 | 98.7031 | 0.167566 | 0.7083 | 0.979877 |
| **W4G64 RTN v4** | **B** | **97** | **98.2363** | **0.162827** | **0.7396** | **0.984625** |

BF16 perplexity is `61.0351` on corpus A and `83.4752` on corpus B. Relative
to W4G128 v1, W4G64 reduces excess NLL by 25.6% on A and 2.6% on B. Corpus A
top-1 agreement falls despite better NLL, so top-1 matching must not be treated
as the only quality objective. NLL/perplexity directly scores the probability of
the ground-truth continuation and is the primary metric here.

These corpora are deliberately small smoke benchmarks, not publication-grade
evaluation. Before claiming production quality, run a larger standard corpus and
task-specific generation evaluation.

## Root cause and fix

The old root checkpoint is unrelated to the official weights and is not
recoverable by changing nibble order. That is a checkpoint provenance problem.

For the newly generated checkpoint, packing and implementation are correct. The
remaining loss is ordinary low-bit quantization error accumulating through 28
decoder layers. W4G128 shares one scale across too many weights and is especially
sensitive to within-group outliers. W4G64 doubles the number of scales, reducing
the range each scale must cover:

- worst direct weight cosine improves from `0.98215669` to `0.98668426`;
- largest per-layer weight MAE improves from `0.00465872` to `0.00399304`;
- checkpoint size grows from about 649 MB to 674 MB.

The W4G64 backend/architecture isolation test reaches `0.99987543` final-logits
cosine against Hugging Face running the identical dequantized weights. Therefore
the observed quality difference comes from quantization, not a backend mismatch.

## Why clipping was rejected

The v2 search allowed clip ratios down to 0.70 and clipped 91.7% of groups; it
increased excess NLL on corpus A from `0.188377` to `0.267498`. The v3 search was
restricted to 0.95--1.00 and improved corpus A, but slightly regressed corpus B.
This is calibration overfitting, so neither artifact is recommended.

Machine-readable reports are in `results/quality_*.json`,
`results/generation_w4g64_rtn_v4.json`, and
`results/hf_validation_w4g64_rtn_v4.json`.
