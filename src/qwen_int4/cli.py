"""Correctness-oriented command line runner with optional cached greedy decode."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .checkpoint import (
    extract_non_quantized_state,
    checkpoint_metadata,
    load_non_quantized_state,
    load_quantized_checkpoint,
    group_size_from_checkpoint,
    packing_from_checkpoint,
)
from .generation import greedy_generate_cached
from .model import QwenInt4ForCausalLM
from .provenance import (
    HF_MODEL_ID,
    HF_MODEL_REVISION,
    HF_TOKENIZER_ID,
    HF_TOKENIZER_REVISION,
)
from .quantization import PackingFormat


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("input IDs must be comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one input ID is required")
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    versioned_checkpoint = (
        project_root / "artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors"
    )
    default_checkpoint = versioned_checkpoint if versioned_checkpoint.is_file() else project_root / "model.safetensors"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--config", default=str(project_root / "configs/qwen2-1.5b"))
    parser.add_argument("--hf-model", default=HF_MODEL_ID)
    parser.add_argument("--hf-revision", default=HF_MODEL_REVISION)
    parser.add_argument("--backend", choices=("torch", "lpu", "triton"), default="torch")
    parser.add_argument(
        "--packing",
        choices=["auto", *[item.value for item in PackingFormat]],
        default="auto",
    )
    parser.add_argument("--device", default=_default_device())
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-ids", type=_parse_ids)
    input_group.add_argument("--prompt")
    parser.add_argument("--tokenizer", default=HF_TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=HF_TOKENIZER_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true", help="Use cached greedy decode when generating tokens.")
    args = parser.parse_args()

    metadata = checkpoint_metadata(args.checkpoint)
    if metadata.get("format_version") != "qwen-int4-v1":
        print(
            "WARNING: this checkpoint's qweight layout has not matched the official "
            "Qwen2-1.5B weights; outputs are diagnostic and not model-correct."
        )

    packing = packing_from_checkpoint(args.checkpoint) if args.packing == "auto" else PackingFormat(args.packing)
    group_size = group_size_from_checkpoint(args.checkpoint)
    config = AutoConfig.from_pretrained(args.config, local_files_only=True)
    model = QwenInt4ForCausalLM(
        config, backend=args.backend, packing=packing, group_size=group_size
    )
    print(f"loading trusted non-quantized tensors from {args.hf_model}@{args.hf_revision}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, revision=args.hf_revision, dtype=torch.bfloat16
    ).eval()
    non_quantized = extract_non_quantized_state(hf_model, model)
    del hf_model
    load_non_quantized_state(model, non_quantized)
    report = load_quantized_checkpoint(model, args.checkpoint)
    model = model.to(args.device).eval()
    print(
        f"loaded {report.loaded_keys} quantized tensors; "
        f"skipped {report.skipped_non_quantized_keys} untrusted checkpoint tensors; "
        f"packing={packing.value}; group_size={group_size}; "
        f"backend={args.backend}; device={args.device}"
    )

    tokenizer = None
    if args.prompt is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.tokenizer_revision
        )
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    else:
        input_ids = torch.tensor([args.input_ids], dtype=torch.long, device=args.device)

    with torch.inference_mode():
        if args.use_cache and args.max_new_tokens:
            input_ids = greedy_generate_cached(model, input_ids, max_new_tokens=args.max_new_tokens)
            logits = model(input_ids)
        else:
            logits = model(input_ids)
            for _ in range(args.max_new_tokens):
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat((input_ids, next_token), dim=-1)
                logits = model(input_ids)

    print(f"input shape: {tuple(input_ids.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"last-token argmax: {logits[:, -1].argmax(dim=-1).tolist()}")
    if tokenizer is not None:
        print(tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
