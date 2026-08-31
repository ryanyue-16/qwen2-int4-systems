"""Compare full frozen-v6 model execution between PyTorch and Triton backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from qwen_int4.checkpoint import extract_non_quantized_state, load_non_quantized_state, load_quantized_checkpoint
from qwen_int4.generation import greedy_generate_cached
from qwen_int4.model import QwenInt4ForCausalLM
from qwen_int4.provenance import HF_MODEL_ID, HF_MODEL_REVISION
from qwen_int4.quantization import PackingFormat
from qwen_int4.validation import tensor_metrics


def _load_model(config, state, checkpoint: str, backend: str) -> QwenInt4ForCausalLM:
    model = QwenInt4ForCausalLM(
        config, backend=backend, packing=PackingFormat.CANONICAL, group_size=64
    )
    load_non_quantized_state(model, state)
    load_quantized_checkpoint(model, checkpoint)
    return model.cuda().eval()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(root / "configs/qwen2-1.5b"))
    parser.add_argument("--max-new-tokens", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-model Triton parity validation requires CUDA")
    config = AutoConfig.from_pretrained(args.config, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, revision=HF_MODEL_REVISION, dtype=torch.bfloat16
    ).eval()
    seed_model = QwenInt4ForCausalLM(config, packing=PackingFormat.CANONICAL, group_size=64)
    state = extract_non_quantized_state(hf_model, seed_model)
    del hf_model, seed_model
    torch_model = _load_model(config, state, args.checkpoint, "torch")
    input_ids = torch.tensor([[9707, 11, 791, 47]], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        torch_logits, torch_cache = torch_model(input_ids, use_cache=True)
        teacher_tokens = []
        torch_steps = []
        for _ in range(args.max_new_tokens):
            token = torch_logits[:, -1].argmax(dim=-1, keepdim=True)
            teacher_tokens.append(token.cpu())
            torch_logits, torch_cache = torch_model(token, past_key_values=torch_cache, use_cache=True)
            torch_steps.append((torch_logits.cpu(), tuple((k.cpu(), v.cpu()) for k, v in torch_cache)))
        torch_logits = torch_model(input_ids).cpu()
        torch_tokens = torch.cat((input_ids.cpu(), *teacher_tokens), dim=1)
    del torch_model
    torch.cuda.empty_cache()
    triton_model = _load_model(config, state, args.checkpoint, "triton")
    with torch.inference_mode():
        triton_logits, triton_cache = triton_model(input_ids, use_cache=True)
        step_diagnostics = []
        for index, teacher_token in enumerate(teacher_tokens):
            triton_logits, triton_cache = triton_model(
                teacher_token.cuda(), past_key_values=triton_cache, use_cache=True
            )
            reference_logits, reference_cache = torch_steps[index]
            cache_errors = [
                max((k.cpu() - rk).abs().max().item(), (v.cpu() - rv).abs().max().item())
                for (k, v), (rk, rv) in zip(triton_cache, reference_cache)
            ]
            step_diagnostics.append({
                "decode_step": index + 1,
                "logits": tensor_metrics(reference_logits, triton_logits.cpu()).to_dict(),
                "first_cache_layer_max_abs": cache_errors[0],
                "worst_cache_layer": int(torch.tensor(cache_errors).argmax()),
                "worst_cache_max_abs": max(cache_errors),
            })
        triton_logits = triton_model(input_ids).cpu()
        triton_tokens = greedy_generate_cached(triton_model, input_ids, max_new_tokens=args.max_new_tokens).cpu()
    metrics = tensor_metrics(torch_logits, triton_logits).to_dict()
    torch.testing.assert_close(triton_logits, torch_logits, rtol=0.05, atol=0.5)
    if not torch.equal(torch_tokens, triton_tokens):
        mismatch = (torch_tokens != triton_tokens).nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(
            "cached greedy generation diverged between PyTorch and Triton backends: "
            + json.dumps({
                "first_mismatch_index": mismatch,
                "torch_tokens": torch_tokens.tolist(),
                "triton_tokens": triton_tokens.tolist(),
                "logits_metrics": metrics,
                "teacher_forced_steps": step_diagnostics,
            })
        )
    print(json.dumps({
        "status": "passed", "input_shape": list(input_ids.shape),
        "logits_metrics": metrics, "cached_greedy_tokens": triton_tokens.cpu().tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
