# HF BF16 numerical validation report

## Setup

- Reference: `Qwen/Qwen2-1.5B`, official Hugging Face BF16 checkpoint.
- Candidate: local `model.safetensors` with trusted non-quantized tensors replaced from the HF reference.
- Input IDs: `[1, 2, 3]`.
- Candidate backend: PyTorch W4A16 reference.
- Accumulation: FP32.
- Full machine-readable report: `results/hf_validation.json` (local, gitignored).

## Result

The checkpoint does **not** currently reproduce the original model.

| Tensor | MAE | Max abs | Cosine |
| --- | ---: | ---: | ---: |
| embedding | 0 | 0 | ~1.0 |
| layer 0 input RMSNorm | 0 | 0 | ~1.0 |
| layer 0 q_proj | 1.250266 | 21.062500 | 0.851625 |
| layer 0 v_proj | 0.372654 | 2.332031 | 0.147151 |
| layer 0 output | 0.675380 | 7.054688 | -0.027353 |
| final logits | 4.450902 | 28.718750 | -0.369142 |

The first material activation divergence is `model.layers.0.self_attn.q_proj`.

## Direct weight comparison

Six representative qweight tensors were dequantized using both candidate formats. Every direct weight cosine was approximately zero:

| Layer | Canonical cosine | Legacy signed-add cosine |
| --- | ---: | ---: |
| layer 0 q_proj | -0.000821 | -0.000649 |
| layer 0 k_proj | 0.000057 | -0.000009 |
| layer 0 o_proj | -0.000496 | -0.000722 |
| layer 0 gate_proj | -0.000166 | -0.000284 |
| layer 0 down_proj | 0.000091 | 0.000124 |
| layer 27 q_proj | -0.000055 | 0.000057 |

Simple `[out, in]`, transposed `[in, out]`, common 8-nibble AWQ orders, and common 8/16/32/64/128/256 O/K tile-axis permutations were also tested without finding meaningful correlation.

## Interpretation

The non-quantized tensors stored in the local checkpoint are not trustworthy base-model parameters. This is consistent with the legacy runtime, which overwrote embedding, Q/K/V biases and norms from Hugging Face every time it started.

After restoring those trusted tensors, divergence begins at the first quantized projection. At least one of the following is therefore true:

1. qweight uses an undocumented LPU-specific reorder not covered by the tested layouts;
2. qweight was generated from a different base/fine-tuned model;
3. the old checkpoint-generation path saved incorrectly packed or incorrectly initialized tensors.

The old quantization/export script is now the highest-value missing artifact. Triton work should remain blocked until the qweight contract is recovered or a new known-correct checkpoint is generated.

