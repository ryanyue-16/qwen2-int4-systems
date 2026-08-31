# Phase D canonical AWQ

## Status

Phase D is complete for correctness and quality gating. Canonical AWQ v6 is
the frozen W4G64 asymmetric candidate for Phase E correctness work. Its model
weights remain on the NVIDIA host and are not committed to this repository.
This decision does not make a CUDA, Triton, LPU, or performance claim.

The selected artifact is identified by SHA-256
`131a020962efc4e236b17fbd7da02f4aa9914e4c88a07b647e8cfe47dcaecab6`.
Its remote checkpoint location and all evidence hashes are recorded in
`artifacts/qwen2-1.5b-w4g64-awq-canonical-v6/manifest.json`.

## Selected recipe and results

All canonical experiments used the pinned Qwen2-1.5B revision and held-out
WikiText-2 test tokens. V6 changed the selected format to W4G64 asymmetric;
v7 was the final targeted ablation and was rejected. No further blind
re-quantization is planned.

| Variant | Format | 2K WikiText-2 PPL ratio | Decision |
| --- | --- | ---: | --- |
| v4 | W4G128 asymmetric | 1.06304 | Rejected |
| v5 | W4G128 asymmetric | 1.07298 | Rejected |
| **v6** | **W4G64 asymmetric** | **1.049220** | **Selected** |
| v7 | W4G64 asymmetric, attention-output objective | 1.054951 | Rejected |

V6 also passed the final three-model Phase C protocol: its full WikiText-2
perplexity ratio was `1.037654`, and its independent Chinese-corpus ratio was
`1.039845`, both below the `1.05` limit. All reported metrics were finite.

## Equivalence and generation review

The final Qwen implementation uses the same SDPA attention behavior as the HF
reference. The final-logits cosine for the selected v6 checkpoint is
`0.9997767806053162`, above the frozen `0.9997` gate. This gate is explicitly
about final logits; it is not a claim that every intermediate tensor is exact.

All 11 deterministic Phase C generation cases were manually reviewed. There
were no empty continuations or replacement characters. Some continuations
diverge from BF16 and include local repetition or logic degradation; this is
recorded rather than hidden. These base-model smoke cases are not an
instruction-following or semantic-equivalence benchmark.

## Phase E handoff

Phase E may begin with correctness-only work against the frozen v6 manifest:
packed-weight decoding, operator equivalence, and fail-closed artifact loading.
Performance optimization, benchmarking, and any Triton/CUDA/LPU claim remain
blocked until their separate documented gates pass.

## Phase E correctness contract

The repository now provides a fail-closed v6 contract validator. It verifies
the frozen manifest identity and Phase E restriction, every local evidence
checksum, and, when given a local copy of the remote checkpoint, its exact
size, SHA-256, and required safetensors metadata. It never downloads, writes,
or regenerates model weights.

Run the repository-only validation with:

```bash
PYTHONPATH=src python scripts/validate_canonical_awq_v6.py
```

On the NVIDIA host, pass the remote checkpoint path explicitly to additionally
validate its bytes and metadata:

```bash
PYTHONPATH=src python scripts/validate_canonical_awq_v6.py \
  --checkpoint /root/autodl-tmp/qwen2-int4-systems/artifacts/qwen2-1.5b-w4g64-awq-canonical-v6-gpu-r1/model.safetensors
```

The Phase E tests cover canonical signed-nibble decoding, per-group asymmetric
scales and zero points, transformed norm-to-linear parameters, malformed
metadata rejection, and reference-equivalent operators at Qwen2 attention
projection shapes. These are artifact and numerical-correctness checks only;
they do not measure or imply kernel performance.
