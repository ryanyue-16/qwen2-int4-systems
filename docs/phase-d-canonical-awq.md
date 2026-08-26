# Phase D canonical AWQ

## Status

Phase D has started in the permitted CPU-testable mode. The implementation
status is `awaiting_gpu_reference`. No checkpoint weight file, reference-quality
comparison, Phase C pass, or GPU-performance claim exists yet.

The implementation follows the official
[AWQ paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf)
and uses the official
[MIT Han Lab implementation](https://github.com/mit-han-lab/llm-awq) as an
algorithm reference. AWQ uses activation statistics to search per-channel
scales, applies function-equivalent parameter transforms, and then performs
weight quantization and clipping. The existing activation-weighted clipping
baseline remains separately named and is not treated as AWQ.

## Implemented CPU foundation

- memory-bounded activation mean, second-moment, and sample collection;
- sequential block calibration with hooks removed after every block;
- Qwen2 mappings for RMSNorm to Q/K/V, V to O, RMSNorm to gate/up, and up to down;
- GQA-aware V-scale expansion across repeated key/value heads;
- per-channel scale search with optional duo scaling and 20-point-grid support;
- candidate evaluation from immutable weight copies;
- state snapshots which restore parameters even when a candidate raises;
- function-equivalent Norm-to-Linear and Linear-to-Linear scale migration;
- function-equivalent Qwen2 SwiGLU up-to-down migration;
- activation-weighted per-output, per-group clipping search;
- asymmetric group-wise INT4 RTN used by search and export;
- canonical signed packing for both weights and per-group zero points;
- PyTorch and LPU functional support for asymmetric zero points;
- versioned canonical export metadata and fail-closed serialization;
- loading of migrated LayerNorm weights and V-projection biases from AWQ artifacts.

## Transformation contracts

For a normalization output consumed by one or more linear layers, channel scale
`s` is migrated as:

```text
norm.weight' = norm.weight / s
linear.weight' = linear.weight * s
```

For two consecutive compatible linear paths:

```text
source.weight' = source.weight / s
source.bias' = source.bias / s
target.weight' = target.weight * s
```

The second relation remains exact for the Qwen2 SwiGLU up path because the up
activation is multiplied elementwise by the unchanged gate activation before
the down projection. For GQA, each V-head scale is repeated in the exact head
order used before the O projection.

## Canonical asymmetric format

The planned artifact is W4G128 asymmetric. Logical codes and zero points are
computed in `[0, 15]`, shifted by eight, and stored as signed INT4 values in the
existing canonical byte order:

```text
dequantized_weight = (signed_code - signed_zero) * scale
```

The new format version is `qwen-int4-awq-v1`. It does not change or migrate the
frozen `qwen-int4-v1` W4G64 RTN checkpoint. A Phase D artifact additionally owns
the migrated LayerNorm parameters and V-projection biases needed by its weight
transform. Loading those tensors from the original HF snapshot would silently
break equivalence, so the loader requires them for the AWQ format.

## Local validation

Run the contract validator:

```bash
PYTHONPATH=src python scripts/validate_awq_canonical.py
```

Run the full CPU suite:

```bash
PYTHONPATH=src python -m pytest
```

The tests cover calibration memory bounds, sequential hook cleanup, scale-search
state safety, Norm/QKV equivalence, V/O GQA equivalence, SwiGLU equivalence,
clipping shapes, asymmetric packing/dequantization, backend agreement, export
metadata, transformed-state loading, and overwrite refusal.

## Remaining gates

The following work remains before Phase D can be completed:

1. connect the primitives into an end-to-end Qwen2 sequential calibration run;
2. run the mature Phase B AWQ reference on NVIDIA Linux;
3. generate a new versioned canonical AWQ checkpoint without replacing RTN v4;
4. perform per-layer weight and activation validation on the full model;
5. run all four Phase C rows on one NVIDIA host;
6. require canonical AWQ to be no worse than W4G64 RTN and target no more than a
   1% relative perplexity gap from the external AWQ reference;
7. freeze the checkpoint contract only after every quality gate passes.

Phase E remains blocked until these items are complete.
