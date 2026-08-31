# Phase C unified quality evaluation

## Status

The user-approved narrowed three-model Phase C v3 evaluation is complete. It
uses BF16, the external W4G128 AWQ reference, and canonical W4G64 AWQ v6 on
the same frozen inputs. The original four-model v1 plan remains preserved as a
historical plan and is not claimed complete because the frozen RTN v4 artifact
was deliberately removed from this remote execution scope.

V3 is the final report because it uses the final SDPA-aligned canonical loader.
It is a full-corpus run, not a development token slice. Its report SHA-256 is
`febfb6681bb168e455ec0f3064b5283bb471d5d5e5568b2b5b2ca3191d32b12c`.

## Frozen inputs and protocol

| Input | SHA-256 |
| --- | --- |
| `data/phase_c/wikitext2_test_full.txt` | `90172ab2c5b6fce8f8a89a438733715ffef0989f220fac42e4fcfdd9757fe1f6` |
| `data/phase_c/chinese_eval_v1.txt` | `590a5dacceafbda5ca01f509c07999ad041d81c51fe64f17edce9075fa238afb` |
| `data/phase_c/generation_prompts_v1.json` | `1494167d293db82128a4769b639eb191c7c92e6cca36db28b23616f758e2e164` |

All rows use the pinned model/tokenizer revision
`8a16abf2848eda07cc5253dec660bf1ce007ad7a`, no added special tokens, a
512-token window, 256-token stride, float64 NLL accumulation, and greedy
32-token decoding. Calibration uses the pinned WikiText-2 train split only.

## Final v3 results

| Model | WikiText-2 PPL | Ratio to BF16 | Chinese PPL | Ratio to BF16 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 10.275594 | 1.000000 | 38.356295 | 1.000000 |
| External AWQ W4G128 | 10.936648 | 1.064332 | 40.637672 | 1.059479 |
| Canonical AWQ v6 W4G64 | 10.662510 | **1.037654** | 39.884600 | **1.039845** |

All numeric metrics were finite. V6 passes the `<= 1.05` ratio gate on both
corpora. The external reference is retained as an executed comparison, but it
does not pass that quality gate.

All 11 v6 generation cases were manually reviewed. No continuation was empty
or contained replacement characters. Token divergence, including some local
repetition and logic degradation relative to BF16, is preserved in the report;
these smoke tests do not establish semantic equivalence.

## Final gate configuration

`configs/phase-c-evaluation-v3.json` freezes the narrowed scope, v6 checkpoint
hash, `<= 1.05` perplexity-ratio gate, and the user-approved final-logits
cosine gate of `>= 0.9997`. Its SHA-256 is
`ffce62b192bea662a8576b21f57836722aee0b25cd95d95b2b0a474935518cef`.
