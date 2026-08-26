"""Evaluate a native compressed-tensors AWQ checkpoint against pinned BF16."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(root / "artifacts/qwen2-1.5b-w4g128-awq-reference-v1"),
    )
    parser.add_argument("--text", default=str(root / "data/wikitext2_test_100rows.txt"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("reference evaluation is restricted to NVIDIA Linux")

    import torch
    import llmcompressor  # noqa: F401 - registers compressed checkpoint loading
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qwen_int4.provenance import (
        HF_MODEL_ID,
        HF_MODEL_REVISION,
        HF_TOKENIZER_REVISION,
    )
    from qwen_int4.validation import sliding_window_language_model_metrics

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reference evaluation")
    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / "runtime-manifest.json").is_file():
        raise FileNotFoundError("the checkpoint has no completed runtime manifest")
    output_path = Path(args.output_json).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; choose a new versioned report"
        )

    text_path = Path(args.text).resolve()
    text = text_path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID, revision=HF_TOKENIZER_REVISION
    )
    input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids[:, : args.max_tokens]
    if input_ids.shape[1] < 2:
        raise ValueError("evaluation text must produce at least two tokens")

    def evaluate(model):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            metrics, top1, target_log_probs = sliding_window_language_model_metrics(
                model,
                input_ids,
                window_size=args.window_size,
                stride=args.stride,
                device="cuda",
            )
        return (
            metrics,
            top1.cpu(),
            target_log_probs.cpu(),
            time.perf_counter() - started,
            torch.cuda.max_memory_allocated(),
        )

    bf16_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, revision=HF_MODEL_REVISION, dtype=torch.bfloat16
    ).to("cuda").eval()
    bf16, bf16_top1, bf16_log_probs, bf16_seconds, bf16_peak = evaluate(bf16_model)
    del bf16_model
    gc.collect()
    torch.cuda.empty_cache()

    awq_model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    awq, awq_top1, awq_log_probs, awq_seconds, awq_peak = evaluate(awq_model)
    finite = all(math.isfinite(value) for value in (*bf16.values(), *awq.values()))
    top1_agreement = (bf16_top1 == awq_top1).float().mean().item()
    target_log_prob_mae = (bf16_log_probs - awq_log_probs).abs().mean().item()

    report = {
        "schema_version": 1,
        "status": "gpu_validated" if finite else "failed_non_finite",
        "hf_model": HF_MODEL_ID,
        "hf_model_revision": HF_MODEL_REVISION,
        "tokenizer_revision": HF_TOKENIZER_REVISION,
        "checkpoint": str(checkpoint),
        "native_format": "compressed-tensors pack_quantized",
        "text": str(text_path),
        "text_sha256": sha256(text_path),
        "tokens": int(input_ids.shape[1]),
        "window_size": args.window_size,
        "stride": args.stride,
        "accumulation_dtype": "float64",
        "bf16": bf16,
        "awq": awq,
        "delta_nll": awq["nll"] - bf16["nll"],
        "perplexity_ratio": awq["perplexity"] / bf16["perplexity"],
        "top1_agreement": top1_agreement,
        "target_log_prob_mae": target_log_prob_mae,
        "all_metrics_finite": finite,
        "runtime_seconds": {"bf16": bf16_seconds, "awq": awq_seconds},
        "peak_gpu_bytes": {"bf16": bf16_peak, "awq": awq_peak},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
