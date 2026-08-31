"""Run the frozen Phase F model-level benchmark on canonical AWQ v6."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from qwen_int4.checkpoint import (
    extract_non_quantized_state,
    load_non_quantized_state,
    load_quantized_checkpoint,
)
from qwen_int4.generation import greedy_generate_cached
from qwen_int4.model import QwenInt4ForCausalLM
from qwen_int4.provenance import HF_MODEL_ID, HF_MODEL_REVISION
from qwen_int4.quantization import PackingFormat


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "minimum_ms": min(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "maximum_ms": max(values),
    }


def timed_cuda(callable_) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop)


def load_model(config, state, checkpoint: str, backend: str) -> QwenInt4ForCausalLM:
    model = QwenInt4ForCausalLM(
        config, backend=backend, packing=PackingFormat.CANONICAL, group_size=64
    )
    load_non_quantized_state(model, state)
    load_quantized_checkpoint(model, checkpoint)
    return model.cuda().eval()


def benchmark_profile(model, profile: dict, warmup: int, iterations: int, seed: int) -> dict:
    batch = profile["batch_size"]
    prompt_tokens = profile["prompt_tokens"]
    new_tokens = profile["new_tokens"]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input_ids = torch.randint(
        0, model.config.vocab_size, (batch, prompt_tokens),
        generator=generator, device="cuda", dtype=torch.long,
    )

    def prefill():
        return model(input_ids, use_cache=True)

    with torch.inference_mode():
        for _ in range(warmup):
            prefill()
        torch.cuda.synchronize()
        prefill_times = [timed_cuda(prefill) for _ in range(iterations)]
        _, cache = prefill()
        decode_ids = torch.zeros((batch, 1), dtype=torch.long, device="cuda")
        decode_mask = torch.ones((batch, prompt_tokens + 1), dtype=torch.bool, device="cuda")

        def decode():
            return model(
                decode_ids, attention_mask=decode_mask,
                past_key_values=cache, use_cache=True,
            )

        for _ in range(warmup):
            decode()
        torch.cuda.synchronize()
        decode_times = [timed_cuda(decode) for _ in range(iterations)]

        def generate():
            return greedy_generate_cached(model, input_ids, max_new_tokens=new_tokens)

        for _ in range(warmup):
            generate()
        torch.cuda.synchronize()
        generation_times = [timed_cuda(generate) for _ in range(iterations)]
        torch.cuda.reset_peak_memory_stats()
        generate()
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()

    generation = summarize(generation_times)
    generated_tokens = batch * new_tokens
    generation["generated_tokens_per_second"] = generated_tokens / (generation["p50_ms"] / 1000.0)
    return {
        "profile": profile,
        "prefill": summarize(prefill_times),
        "single_token_decode": summarize(decode_times),
        "cached_generation": generation,
        "peak_allocated_memory_bytes": peak_memory,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(root / "configs/phase-f-model-benchmark-v1.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint, config_path, output = Path(args.checkpoint), Path(args.config), Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "correctness_gates_passed_benchmark_authorized":
        raise ValueError("benchmark config is not authorized by the correctness gates")
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 does not match the frozen benchmark config")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase F model benchmark requires CUDA")

    model_config = AutoConfig.from_pretrained(str(root / "configs/qwen2-1.5b"), local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, revision=HF_MODEL_REVISION, dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    skeleton = QwenInt4ForCausalLM(
        model_config, packing=PackingFormat.CANONICAL, group_size=64
    )
    state = extract_non_quantized_state(hf_model, skeleton)
    del hf_model, skeleton

    results = []
    for backend_index, backend in enumerate(config["backends"]):
        model = load_model(model_config, state, str(checkpoint), backend)
        backend_profiles = [
            benchmark_profile(
                model, profile, config["warmup_iterations"],
                config["measurement_iterations"], config["seed"] + index,
            )
            for index, profile in enumerate(config["profiles"])
        ]
        results.append({"backend": backend, "profiles": backend_profiles})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "phase": "F",
        "status": "completed",
        "checkpoint_sha256": config["checkpoint_sha256"],
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "config": str(config_path),
        "results": results,
        "limitations": [
            "Results apply only to the frozen profiles and recorded environment.",
            "The Triton backend is hybrid and includes an explicit FP32 PyTorch fallback.",
            "The PyTorch backend is a correctness reference, not a production baseline.",
            "No real-LPU or production-service performance claim is made."
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
