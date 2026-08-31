# Phase F completion report

## Outcome

Phase F is complete for the frozen scope. The repository now has a
correctness-tested KV cache, separate prefill and decode paths, left-padded
batching, request reorder/removal, explicit cache release, cached greedy
generation, hybrid Triton model dispatch, an 11-case correctness matrix, and a
versioned model benchmark protocol.

The hybrid backend is deliberately not called all-Triton. FP16/BF16 canonical
W4G64 linears use Triton with FP32 accumulation; FP32 SwiGLU down projections
use an explicit PyTorch reference fallback. The expanded correctness matrix
matched cached greedy tokens in all 11 cases and achieved a minimum prompt
logits cosine of `0.999700665473938`.

## Formal v2 benchmark

The formal report uses five warmup iterations and 30 measured iterations per
profile on one RTX 4090 with PyTorch `2.12.0+cu130` and CUDA `13.0`.

| Profile | Metric | PyTorch reference p50 | Hybrid p50 | Speedup |
| --- | --- | ---: | ---: | ---: |
| B1, prompt 16 | Prefill | 90.920 ms | 45.960 ms | 1.98x |
| B1, prompt 16 | Decode | 84.027 ms | 40.036 ms | 2.10x |
| B1, prompt 16 | Cached generation | 425.452 ms | 203.285 ms | 2.09x |
| B1, prompt 128 | Prefill | 97.352 ms | 89.830 ms | 1.08x |
| B1, prompt 128 | Decode | 85.882 ms | 40.218 ms | 2.14x |
| B1, prompt 128 | Cached generation | 428.600 ms | 247.261 ms | 1.73x |
| B4, prompt 16 | Prefill | 91.906 ms | 63.435 ms | 1.45x |
| B4, prompt 16 | Decode | 85.416 ms | 40.830 ms | 2.09x |
| B4, prompt 16 | Cached generation | 426.358 ms | 223.908 ms | 1.90x |

The versioned local summary is
`results/phase_f_model_benchmark_summary_v2.json`. It records the SHA-256 of
the complete remote report, which contains p50, p95, p99, minimum, maximum,
mean, generated tokens per second, and peak allocated memory for every profile.

## Interpretation

These are reproducible model-level measurements for the listed synthetic token
profiles, frozen v6 checkpoint, software versions, and single RTX 4090. The
PyTorch backend is a correctness-oriented dequantize-then-matmul reference, not
a production serving baseline. The results do not establish production service
latency, multi-GPU scaling, an all-Triton model, or real-LPU performance.
