"""Memory-bounded held-out NLL/PPL comparison for BF16 and INT4 models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from qwen_int4.checkpoint import (
    extract_non_quantized_state,
    load_non_quantized_state,
    load_quantized_checkpoint,
    group_size_from_checkpoint,
    packing_from_checkpoint,
)
from qwen_int4.model import QwenInt4ForCausalLM
from qwen_int4.provenance import HF_MODEL_ID, HF_MODEL_REVISION, HF_TOKENIZER_REVISION
from qwen_int4.validation import sliding_window_language_model_metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default=HF_MODEL_ID)
    parser.add_argument("--hf-revision", default=HF_MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=HF_TOKENIZER_REVISION)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default=str(root / "data/evaluation_corpus.txt"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--backend", choices=("torch", "lpu"), default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output_json)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --overwrite explicitly"
        )

    text = Path(args.text).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.max_tokens]
    if input_ids.shape[1] < 2:
        raise ValueError("evaluation text must produce at least two tokens")
    print(f"evaluation tokens: {input_ids.shape[1]}")

    config = AutoConfig.from_pretrained(str(root / "configs/qwen2-1.5b"), local_files_only=True)
    skeleton = QwenInt4ForCausalLM(config)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(args.device).eval()
    non_quantized = extract_non_quantized_state(hf_model, skeleton)
    del skeleton
    with torch.inference_mode():
        bf16_metrics, bf16_top1, bf16_target_log_probs = (
            sliding_window_language_model_metrics(
                hf_model,
                input_ids,
                window_size=args.window_size,
                stride=args.stride,
                device=args.device,
            )
        )
    print(f"BF16: nll={bf16_metrics['nll']:.6f}, ppl={bf16_metrics['perplexity']:.4f}")
    del hf_model
    gc.collect()

    packing = packing_from_checkpoint(args.checkpoint)
    group_size = group_size_from_checkpoint(args.checkpoint)
    quant_model = QwenInt4ForCausalLM(
        config, backend=args.backend, packing=packing, group_size=group_size
    )
    load_non_quantized_state(quant_model, non_quantized)
    load_quantized_checkpoint(quant_model, args.checkpoint)
    quant_model = quant_model.to(args.device).eval()
    with torch.inference_mode():
        quant_metrics, quant_top1, quant_target_log_probs = (
            sliding_window_language_model_metrics(
                quant_model,
                input_ids,
                window_size=args.window_size,
                stride=args.stride,
                device=args.device,
            )
        )
    top1_agreement = (bf16_top1 == quant_top1).to(torch.float32).mean().item()
    target_log_prob_mae = (
        bf16_target_log_probs - quant_target_log_probs
    ).abs().mean().item()
    print(f"INT4: nll={quant_metrics['nll']:.6f}, ppl={quant_metrics['perplexity']:.4f}")
    print(
        f"delta_nll={quant_metrics['nll'] - bf16_metrics['nll']:.6f}, "
        f"top1_agreement={top1_agreement:.4f}, "
        f"target_log_prob_mae={target_log_prob_mae:.6f}"
    )

    output = {
        "hf_model": args.hf_model,
        "hf_model_revision": args.hf_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "text": str(Path(args.text)),
        "tokens": input_ids.shape[1],
        "checkpoint": args.checkpoint,
        "packing": packing.value,
        "group_size": group_size,
        "bf16": bf16_metrics,
        "int4": quant_metrics,
        "delta_nll": quant_metrics["nll"] - bf16_metrics["nll"],
        "perplexity_ratio": quant_metrics["perplexity"] / bf16_metrics["perplexity"],
        "top1_agreement": top1_agreement,
        "target_log_prob_mae": target_log_prob_mae,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
