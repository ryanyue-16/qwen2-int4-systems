# Symmetric RTN W4G128 v1 validation

## Artifact

- Checkpoint: `artifacts/qwen2-1.5b-w4g128-rtn-v1/model.safetensors`
- Size: approximately 649 MB.
- Source: official `Qwen/Qwen2-1.5B` BF16 checkpoint.
- Quantization: symmetric round-to-nearest, group size 128, code range `[-7,7]`.
- Packing: row-major `[out, in]`; K-even in low nibble and K-odd in high nibble.
- Metadata: embedded in safetensors; format version `qwen-int4-v1`.

## Direct weight validation

All 196 linear layers are quantized and immediately dequantized during export.

- Worst weight cosine: `0.98215669`.
- Largest per-layer mean absolute error: `0.00465872`.
- Representative layer cosine values are approximately `0.99`.

These results confirm that pack → save → load → unpack → scale reconstructs the intended RTN weights.

## Architecture/backend isolation

The same dequantized weights were evaluated through:

1. the official Hugging Face Qwen2 architecture;
2. this repository's Qwen2 model and PyTorch reference backend.

Final logits comparison:

- cosine: `0.99986672`;
- MAE: `0.02436388`;
- max absolute error: `0.25`.

This small difference is consistent with BF16 versus FP32 accumulation and operation-order differences. It confirms the custom architecture/backend is using the new checkpoint correctly.

## BF16 versus RTN INT4

For input IDs `[1,2,3]`, official BF16 versus the W4G128 model gives:

- logits cosine: `0.82351202`;
- logits MAE: `1.17303538`;
- max absolute error: `17.7734375`.

The quantization implementation is correct, but a single short-input logits comparison suggests naïve RTN may lose too much model-level fidelity. This is not yet a quality benchmark. The next decision must be based on perplexity and representative prompt evaluation; if degradation is excessive, introduce clipping/search or AWQ-style activation-aware scaling while keeping the v1 RTN checkpoint as the numerical baseline.

Machine-readable results are stored locally in `results/hf_validation_rtn_v1.json`.

