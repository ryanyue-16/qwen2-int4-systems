"""Run the frozen Phase F torch/hybrid correctness matrix on canonical v6."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from qwen_int4.checkpoint import (
    extract_non_quantized_state,
    load_non_quantized_state,
    load_quantized_checkpoint,
)
from qwen_int4.generation import greedy_generate_cached
from qwen_int4.model import QwenInt4ForCausalLM
from qwen_int4.provenance import HF_MODEL_ID, HF_MODEL_REVISION, HF_TOKENIZER_REVISION
from qwen_int4.quantization import PackingFormat
from qwen_int4.validation import tensor_metrics


V6_SHA256 = "131a020962efc4e236b17fbd7da02f4aa9914e4c88a07b647e8cfe47dcaecab6"


def load_model(config, state, checkpoint: str, backend: str) -> QwenInt4ForCausalLM:
    model = QwenInt4ForCausalLM(
        config, backend=backend, packing=PackingFormat.CANONICAL, group_size=64
    )
    load_non_quantized_state(model, state)
    load_quantized_checkpoint(model, checkpoint)
    return model.cuda().eval()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", default=str(root / "data/phase_c/generation_prompts_v1.json"))
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint, output = Path(args.checkpoint), Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256_file(checkpoint) != V6_SHA256:
        raise ValueError("checkpoint SHA-256 does not match frozen canonical v6")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID, revision=HF_TOKENIZER_REVISION, local_files_only=True
    )
    config = AutoConfig.from_pretrained(str(root / "configs/qwen2-1.5b"), local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, revision=HF_MODEL_REVISION, dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    skeleton = QwenInt4ForCausalLM(config, packing=PackingFormat.CANONICAL, group_size=64)
    state = extract_non_quantized_state(hf_model, skeleton)
    del hf_model, skeleton

    torch_model = load_model(config, state, str(checkpoint), "torch")
    references = []
    with torch.inference_mode():
        for item in prompts:
            ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.cuda()
            references.append({
                "item": item,
                "ids": ids.cpu(),
                "logits": torch_model(ids).cpu(),
                "tokens": greedy_generate_cached(
                    torch_model, ids, max_new_tokens=args.max_new_tokens
                ).cpu(),
            })
    del torch_model
    gc.collect()
    torch.cuda.empty_cache()

    triton_model = load_model(config, state, str(checkpoint), "triton")
    cases = []
    with torch.inference_mode():
        for reference in references:
            ids = reference["ids"].cuda()
            logits = triton_model(ids).cpu()
            tokens = greedy_generate_cached(
                triton_model, ids, max_new_tokens=args.max_new_tokens
            ).cpu()
            metrics = tensor_metrics(reference["logits"], logits).to_dict()
            cases.append({
                "id": reference["item"]["id"],
                "category": reference["item"]["category"],
                "prompt_tokens": ids.shape[1],
                "logits_metrics": metrics,
                "generation_tokens_match": bool(torch.equal(reference["tokens"], tokens)),
                "torch_continuation": reference["tokens"][0, ids.shape[1]:].tolist(),
                "hybrid_continuation": tokens[0, ids.shape[1]:].tolist(),
            })
    exact = sum(case["generation_tokens_match"] for case in cases)
    minimum_cosine = min(case["logits_metrics"]["cosine"] for case in cases)
    passed = exact == len(cases) and minimum_cosine >= 0.9997
    report = {
        "schema_version": 1,
        "phase": "F",
        "status": "passed" if passed else "failed",
        "checkpoint_sha256": V6_SHA256,
        "max_new_tokens": args.max_new_tokens,
        "cases_total": len(cases),
        "generation_exact_cases": exact,
        "minimum_logits_cosine": minimum_cosine,
        "cases": cases,
        "limitations": [
            "This is a deterministic correctness matrix, not a performance benchmark.",
            "The hybrid backend retains the explicit FP32 PyTorch fallback."
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise AssertionError("Phase F correctness matrix failed; inspect the versioned report")


if __name__ == "__main__":
    main()
