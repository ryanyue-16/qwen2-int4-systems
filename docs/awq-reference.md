# Mature AWQ reference

## Decision

Phase B uses [LLM Compressor 0.13.0](https://pypi.org/project/llmcompressor/0.13.0/)
from the vLLM project as the external AWQ oracle. The selection was made from
official documentation, repositories, and release metadata only.

LLM Compressor is preferred over AutoAWQ for this project because the current
[Transformers AWQ documentation](https://huggingface.co/docs/transformers/quantization/awq)
warns that installing AutoAWQ downgrades Transformers to 4.47.1. The official
[MIT Han Lab AWQ repository](https://github.com/mit-han-lab/llm-awq) remains an
important research reference, but LLM Compressor provides the maintained
compression and vLLM deployment path used here.

## Frozen reference contract

| Property | Phase B value |
| --- | --- |
| Model | `Qwen/Qwen2-1.5B` at revision `8a16abf...` |
| Implementation | `llmcompressor==0.13.0` |
| Scheme | `W4A16_ASYM` |
| Group size | 128 |
| Linear targets | All `Linear` modules except `lm_head` |
| AWQ scaling | Duo scaling enabled, 20-point scale search default |
| Calibration | 256 deterministic WikiText-2 raw train samples, 512 tokens maximum |
| Pipeline | Sequential onloading |
| Export | Native `compressed-tensors` `pack_quantized` |
| Serving validation | Native checkpoint loaded by `vllm==0.27.1` in a separate runtime environment |
| Current status | Executed; not selected because final Phase C PPL ratios exceed `1.05` |

The official [AWQ recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/awq/)
pairs `AWQModifier` with `QuantizationModifier`, uses `W4A16_ASYM`, and leaves
`lm_head` in higher precision. The tagged
[Qwen2 mappings](https://github.com/vllm-project/llm-compressor/blob/0.13.0/src/llmcompressor/modifiers/transform/awq/mappings.py)
cover input LayerNorm to Q/K/V, V to O, post-attention LayerNorm to gate/up, and
up to down projections. This includes Qwen2 GQA and SwiGLU projection paths.

The calibration source is `Salesforce/wikitext`, `wikitext-2-raw-v1`, train
split at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. The official
[dataset snapshot](https://huggingface.co/datasets/Salesforce/wikitext/commit/b08601e04326c79dfdd32d625aee71d232d685c3)
contains distinct train, validation, and test files. Phase B calibration uses
train only; quality evaluation uses test only.

## Native format and runtime

The external format is deliberately preserved. LLM Compressor serializes
quantized weights through `compressed-tensors`; the Phase B artifact records
`pack_quantized` rather than this repository's canonical nibble contract. Scale
orientation, zero-point tensor orientation, and packed tensor orientation must
be inspected from the generated artifact before any converter is written.

No equality with `qwen-int4-v1` is assumed. The official
[checkpoint conversion documentation](https://docs.vllm.ai/projects/llm-compressor/en/stable/guides/entrypoints/convert/)
states that converting AutoAWQ to compressed-tensors requires unpacking,
bit-layout reordering, and a shift to signed integers. This is direct evidence
that the names “AWQ” and “INT4” do not define a universal packing layout.

Transformers can load compressed-tensors checkpoints for correctness testing,
but it decompresses them for forward execution and is not the optimized
backend. Native serving validation therefore uses vLLM. Kernel selection is a
vLLM runtime decision based on the installed build and GPU; this repository does
not claim a specific GEMM/GEMV kernel until the NVIDIA run records it. The
[vLLM package](https://pypi.org/project/vllm/0.27.1/) supports Qwen models,
AWQ, and compressed-tensors.

LLM Compressor and vLLM are deliberately installed in separate environments.
`llmcompressor==0.13.0` requires `compressed-tensors==0.18.0`, whereas
`vllm==0.27.1` requires `compressed-tensors==0.17.0`; pip cannot resolve both
strict requirements in one environment. The reference environment is therefore
reserved for quantization and Transformers correctness evaluation, while the
vLLM environment is reserved for native serving smoke tests.

The selected LLM Compressor release requires Linux and a compatible CUDA-enabled
PyTorch build. Its tagged
[dependency metadata](https://github.com/vllm-project/llm-compressor/blob/0.13.0/setup.py)
allows PyTorch 2.10 through 2.13 and Transformers 5.9 through 5.14.1. The isolated
environment pins versions inside those ranges. A Turing-or-newer GPU is the
minimum target for W4A16 deployment; Ampere or newer is preferred for the first
validation run.

## NVIDIA execution

Create the isolated environment on a Linux NVIDIA host:

```bash
conda env create -f environments/awq-reference-cu12.yml
conda activate qwen-awq-reference
python -m pip check
```

Validate the frozen plan without loading GPU packages:

```bash
PYTHONPATH=src python scripts/quantize_awq_reference.py --validate-only
```

Create the versioned native checkpoint:

```bash
PYTHONPATH=src python scripts/quantize_awq_reference.py
```

Run the initial 2K-token correctness comparison without overwriting an existing
report:

```bash
PYTHONPATH=src python scripts/evaluate_awq_reference.py \
  --output-json results/quality_awq_reference_v1_wikitext2_2k.json
```

Run a native-format vLLM generation smoke test:

```bash
conda env create -f environments/vllm-smoke-cu13.yml
conda activate qwen-vllm-smoke
python -m vllm.entrypoints.openai.api_server \
  --model artifacts/qwen2-1.5b-w4g128-awq-reference-v1 \
  --dtype bfloat16 \
  --seed 42
```

After quantization, preserve the generated `runtime-manifest.json`, capture
`python -m pip freeze`, and record the vLLM startup log. The execution evidence
exists on the NVIDIA host. The reference remains an executed comparison rather
than the selected artifact because its final Phase C PPL ratios are `1.064332`
on WikiText-2 and `1.059479` on the independent Chinese corpus, both above the
project's `<= 1.05` gate. No performance conclusion follows from this result.
