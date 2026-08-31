# INT4 quality analysis

## Current decision

The selected canonical AWQ artifact is W4G64 asymmetric v6. The frozen RTN v4
artifact remains a historical baseline and was not regenerated or overwritten.
V6 is selected because it is the only canonical AWQ candidate that passed both
the held-out 2K gate and the final full-corpus Phase C gates.

| Evaluation | BF16 PPL | V6 PPL | V6 ratio |
| --- | ---: | ---: | ---: |
| WikiText-2, 2K tokens | 11.800438 | 12.381258 | 1.049220 |
| WikiText-2, full Phase C | 10.275594 | 10.662510 | 1.037654 |
| Independent Chinese corpus, full Phase C | 38.356295 | 39.884600 | 1.039845 |

All reported metrics are finite. V6 final-logits cosine against the HF
dequantized reference is `0.9997767806053162`, above the frozen `0.9997` gate.
The final loader uses SDPA to match the HF reference attention behavior.

## Candidate comparison

The W4G128 v4 and v5 recipes did not meet the 2K `<= 1.05` ratio gate. V7 was
a final targeted W4G64 attention-output ablation, but its ratio was `1.054951`.
It was rejected. The external W4G128 AWQ reference completed the same full
Phase C protocol but scored `1.064332` on WikiText-2 and `1.059479` on the
Chinese corpus, also above the gate.

This conclusion is based on recorded experiments, not extrapolation; no more
blind quantization variants are authorized by the quality decision.

## Generation limitations

The 11 deterministic generation cases contain no empty output or replacement
characters for v6. They do show token-level divergence and isolated local
repetition or logic degradation relative to BF16. This is retained in
`results/phase_c_evaluation_v3_gpu_run1.json`. The prompts test regressions of
a base model, not instruction following or semantic equivalence.

## Scope of the decision

The v6 manifest permits Phase E correctness work only: decoding, loading, and
operator-equivalence validation. It does not establish a CUDA kernel, Triton
kernel, true LPU runtime, latency, throughput, or production-quality claim.
