"""Run the frozen Phase C quality protocol across available model variants."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from qwen_int4.checkpoint import (
    extract_non_quantized_state,
    group_size_from_checkpoint,
    load_non_quantized_state,
    load_quantized_checkpoint,
    packing_from_checkpoint,
)
from qwen_int4.evaluation import (
    all_finite,
    generation_regression,
    load_and_validate_phase_c_config,
    model_availability,
    sha256_file,
)
from qwen_int4.model import QwenInt4ForCausalLM
from qwen_int4.validation import sliding_window_language_model_metrics


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(root / "configs/phase-c-evaluation-v1.json")
    )
    parser.add_argument(
        "--models",
        default="bf16,w4g64_rtn_v4",
        help="Comma-separated model IDs from the Phase C config; BF16 is mandatory.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Development-only token cap. Omit for a formal full-corpus report.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def logits_from(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    output = (
        model(input_ids, use_cache=False)
        if "use_cache" in signature(model.forward).parameters
        else model(input_ids)
    )
    return output.logits if hasattr(output, "logits") else output


def greedy_continuation(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    device: str,
) -> list[int]:
    input_ids = prompt_ids.to(device)
    prompt_length = input_ids.shape[1]
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = logits_from(model, input_ids)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
    return input_ids[0, prompt_length:].detach().cpu().tolist()


def peak_memory_bytes(device: str) -> tuple[int, str]:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated(device), "torch_cuda_max_memory_allocated"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        peak *= 1024
    return int(peak), "process_peak_rss"


def reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def evaluate_model(
    model: torch.nn.Module,
    *,
    corpus_ids: dict[str, torch.Tensor],
    prompt_ids: dict[str, torch.Tensor],
    prompts: list[dict[str, str]],
    tokenizer,
    scoring: dict[str, Any],
    generation: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus_results: dict[str, Any] = {}
    for corpus_id, input_ids in corpus_ids.items():
        reset_peak_memory(device)
        started = time.perf_counter()
        with torch.inference_mode():
            metrics, top1, target_log_probs = sliding_window_language_model_metrics(
                model,
                input_ids,
                window_size=scoring["window_size"],
                stride=scoring["stride"],
                device=device,
            )
        elapsed = time.perf_counter() - started
        peak_bytes, peak_source = peak_memory_bytes(device)
        corpus_results[corpus_id] = {
            "metrics": metrics,
            "all_metrics_finite": all_finite(metrics),
            "runtime_seconds": elapsed,
            "peak_memory_bytes": peak_bytes,
            "peak_memory_source": peak_source,
            "top1": top1,
            "target_log_probs": target_log_probs,
        }

    generation_results: dict[str, Any] = {}
    for item in prompts:
        started = time.perf_counter()
        token_ids = greedy_continuation(
            model,
            prompt_ids[item["id"]],
            max_new_tokens=generation["max_new_tokens"],
            device=device,
        )
        text = tokenizer.decode(token_ids)
        generation_results[item["id"]] = {
            "category": item["category"],
            "token_ids": token_ids,
            "continuation": text,
            "runtime_seconds": time.perf_counter() - started,
            "contains_replacement_character": "\ufffd" in text,
        }
    return corpus_results, generation_results


def load_canonical_model(
    checkpoint: Path,
    local_config: AutoConfig,
    non_quantized: dict[str, torch.Tensor],
    device: str,
) -> QwenInt4ForCausalLM:
    model = QwenInt4ForCausalLM(
        local_config,
        backend="torch",
        packing=packing_from_checkpoint(checkpoint),
        group_size=group_size_from_checkpoint(checkpoint),
    )
    load_non_quantized_state(model, non_quantized)
    load_quantized_checkpoint(model, checkpoint)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_and_validate_phase_c_config(config_path, root)
    statuses = model_availability(config, root)
    requested = [value.strip() for value in args.models.split(",") if value.strip()]
    model_configs = {model["id"]: model for model in config["models"]}
    unknown = set(requested) - set(model_configs)
    if unknown:
        raise ValueError(f"unknown model IDs: {sorted(unknown)}")
    if "bf16" not in requested:
        raise ValueError("BF16 must be included as the comparison anchor")
    unavailable = {
        model_id: statuses[model_id]
        for model_id in requested
        if statuses[model_id] != "ready"
    }
    plan = {
        "evaluation_id": config["evaluation_id"],
        "models": requested,
        "model_status": {model_id: statuses[model_id] for model_id in requested},
        "corpora": [corpus["id"] for corpus in config["corpora"]],
        "generation_cases": len(json.loads((root / config["generation"]["path"]).read_text())),
        "formal_full_corpus": args.max_tokens is None,
        "device": args.device,
    }
    if args.validate_only:
        print(json.dumps(plan, indent=2))
        return
    if unavailable:
        raise RuntimeError(f"requested models are not ready: {unavailable}")
    if not args.output_json:
        raise ValueError("--output-json is required for an evaluation run")
    output_path = Path(args.output_json).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    model_identity = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_identity["id"],
        revision=model_identity["tokenizer_revision"],
        local_files_only=args.local_files_only,
    )
    corpus_ids: dict[str, torch.Tensor] = {}
    for corpus in config["corpora"]:
        text = (root / corpus["path"]).read_text(encoding="utf-8")
        ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=config["scoring"]["add_special_tokens"],
        ).input_ids
        if args.max_tokens is not None:
            ids = ids[:, : args.max_tokens]
        if ids.shape[1] < 2:
            raise ValueError(f"corpus {corpus['id']} produced fewer than two tokens")
        corpus_ids[corpus["id"]] = ids

    prompts = json.loads((root / config["generation"]["path"]).read_text(encoding="utf-8"))
    prompt_ids = {
        item["id"]: tokenizer(
            item["prompt"],
            return_tensors="pt",
            add_special_tokens=config["generation"]["add_special_tokens"],
        ).input_ids
        for item in prompts
    }

    local_config = AutoConfig.from_pretrained(
        root / "configs/qwen2-1.5b", local_files_only=True
    )
    needs_canonical = any(
        model_configs[model_id]["kind"] == "qwen_int4_canonical"
        for model_id in requested
    )
    skeleton = QwenInt4ForCausalLM(local_config) if needs_canonical else None
    bf16_model = AutoModelForCausalLM.from_pretrained(
        model_identity["id"],
        revision=model_identity["revision"],
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(args.device).eval()
    non_quantized = (
        extract_non_quantized_state(bf16_model, skeleton) if skeleton is not None else {}
    )
    del skeleton

    results: dict[str, Any] = {}
    bf16_corpora, bf16_generation = evaluate_model(
        bf16_model,
        corpus_ids=corpus_ids,
        prompt_ids=prompt_ids,
        prompts=prompts,
        tokenizer=tokenizer,
        scoring=config["scoring"],
        generation=config["generation"],
        device=args.device,
    )
    results["bf16"] = {"corpora": bf16_corpora, "generation": bf16_generation}
    del bf16_model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    for model_id in requested:
        if model_id == "bf16":
            continue
        model_config = model_configs[model_id]
        checkpoint = (root / model_config["checkpoint"]).resolve()
        if model_config["kind"] == "qwen_int4_canonical":
            model = load_canonical_model(
                checkpoint, local_config, non_quantized, args.device
            )
        elif model_config["kind"] == "compressed_tensors":
            import llmcompressor  # noqa: F401 - registers compressed loading

            model = AutoModelForCausalLM.from_pretrained(
                checkpoint, dtype=torch.bfloat16, local_files_only=True
            ).to(args.device).eval()
        else:
            raise ValueError(f"unsupported model kind: {model_config['kind']}")
        corpora, generations = evaluate_model(
            model,
            corpus_ids=corpus_ids,
            prompt_ids=prompt_ids,
            prompts=prompts,
            tokenizer=tokenizer,
            scoring=config["scoring"],
            generation=config["generation"],
            device=args.device,
        )
        for corpus_id, values in corpora.items():
            reference = bf16_corpora[corpus_id]
            values["comparison_to_bf16"] = {
                "delta_nll": values["metrics"]["nll"] - reference["metrics"]["nll"],
                "perplexity_ratio": (
                    values["metrics"]["perplexity"]
                    / reference["metrics"]["perplexity"]
                ),
                "top1_agreement": (
                    values["top1"] == reference["top1"]
                ).float().mean().item(),
                "target_log_prob_mae": (
                    values["target_log_probs"] - reference["target_log_probs"]
                ).abs().mean().item(),
            }
        for prompt_id, values in generations.items():
            values["comparison_to_bf16"] = generation_regression(
                bf16_generation[prompt_id]["token_ids"], values["token_ids"]
            )
            values["manual_collapse_review_required"] = True
        results[model_id] = {"corpora": corpora, "generation": generations}
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    for model_result in results.values():
        for corpus_result in model_result["corpora"].values():
            corpus_result.pop("top1")
            corpus_result.pop("target_log_probs")

    report = {
        "schema_version": 1,
        "evaluation_id": config["evaluation_id"],
        "status": "complete" if set(requested) == set(model_configs) else "partial",
        "formal_full_corpus": args.max_tokens is None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "model_identity": model_identity,
        "tokenization": {
            "add_special_tokens": config["scoring"]["add_special_tokens"],
            "corpus_token_counts": {
                key: int(value.shape[1]) for key, value in corpus_ids.items()
            },
        },
        "scoring": config["scoring"],
        "generation_config": config["generation"],
        "device": args.device,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "results": results,
    }
    if any(
        not corpus["all_metrics_finite"]
        for model in results.values()
        for corpus in model["corpora"].values()
    ):
        report["status"] = "failed_non_finite"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
