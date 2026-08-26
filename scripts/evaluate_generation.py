"""Compare deterministic BF16 and INT4 greedy continuations."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from qwen_int4.checkpoint import (
    extract_non_quantized_state,
    group_size_from_checkpoint,
    load_non_quantized_state,
    load_quantized_checkpoint,
    packing_from_checkpoint,
)
from qwen_int4.model import QwenInt4ForCausalLM
from qwen_int4.provenance import HF_MODEL_ID, HF_MODEL_REVISION, HF_TOKENIZER_REVISION


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        length += 1
    return length


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default=HF_MODEL_ID)
    parser.add_argument("--hf-revision", default=HF_MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=HF_TOKENIZER_REVISION)
    parser.add_argument(
        "--checkpoint",
        default=str(root / "artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors"),
    )
    parser.add_argument("--prompts", default=str(root / "data/generation_prompts.json"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-json", default=str(root / "results/generation_w4g64_rtn_v4.json")
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output_json)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --overwrite explicitly"
        )

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(
        str(root / "configs/qwen2-1.5b"), local_files_only=True
    )
    skeleton = QwenInt4ForCausalLM(config)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(args.device).eval()
    non_quantized = extract_non_quantized_state(hf_model, skeleton)
    del skeleton

    results: dict[str, dict] = {}
    print("Generating BF16 continuations")
    for item in prompts:
        encoded = tokenizer(item["prompt"], return_tensors="pt")
        input_ids = encoded.input_ids.to(args.device)
        attention_mask = encoded.attention_mask.to(args.device)
        with torch.inference_mode():
            output_ids = hf_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
        results[item["id"]] = {
            "prompt": item["prompt"],
            "bf16_token_ids": continuation,
            "bf16_continuation": tokenizer.decode(continuation),
        }
    del hf_model
    gc.collect()

    packing = packing_from_checkpoint(args.checkpoint)
    group_size = group_size_from_checkpoint(args.checkpoint)
    quant_model = QwenInt4ForCausalLM(
        config, backend="torch", packing=packing, group_size=group_size
    )
    load_non_quantized_state(quant_model, non_quantized)
    load_quantized_checkpoint(quant_model, args.checkpoint)
    quant_model = quant_model.to(args.device).eval()

    print("Generating INT4 continuations")
    for item in prompts:
        input_ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.to(args.device)
        prompt_length = input_ids.shape[1]
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                logits = quant_model(input_ids)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat((input_ids, next_token), dim=-1)
        continuation = input_ids[0, prompt_length:].detach().cpu().tolist()
        reference = results[item["id"]]["bf16_token_ids"]
        position_matches = sum(a == b for a, b in zip(reference, continuation))
        results[item["id"]].update(
            {
                "int4_token_ids": continuation,
                "int4_continuation": tokenizer.decode(continuation),
                "common_prefix_tokens": common_prefix_length(reference, continuation),
                "position_agreement": position_matches / args.max_new_tokens,
            }
        )
        print(
            f"{item['id']}: agreement={position_matches / args.max_new_tokens:.3f}, "
            f"common_prefix={results[item['id']]['common_prefix_tokens']}"
        )

    output = {
        "hf_model": args.hf_model,
        "hf_model_revision": args.hf_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "checkpoint": args.checkpoint,
        "max_new_tokens": args.max_new_tokens,
        "mean_position_agreement": sum(
            item["position_agreement"] for item in results.values()
        ) / len(results),
        "cases": list(results.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
