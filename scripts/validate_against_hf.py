"""Compare checkpoint weights and activations against the original HF BF16 model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from qwen_int4.checkpoint import (
    extract_non_quantized_state,
    load_non_quantized_state,
    load_quantized_checkpoint,
    group_size_from_checkpoint,
    packing_from_checkpoint,
)
from qwen_int4.export import AWQ_TRANSFORMED_SUFFIXES
from qwen_int4.model import QwenInt4ForCausalLM
from qwen_int4.provenance import HF_MODEL_ID, HF_MODEL_REVISION, HF_TOKENIZER_REVISION
from qwen_int4.quantization import PackingFormat, dequantize_groupwise
from qwen_int4.validation import ActivationRecorder, qwen_module_names, tensor_metrics


DEFAULT_WEIGHT_LAYERS = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.27.self_attn.q_proj",
)


def parse_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("input IDs must be comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one token ID is required")
    return result


def weight_format_comparison(hf_model, checkpoint: str, names: tuple[str, ...]) -> dict:
    modules = dict(hf_model.named_modules())
    results = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for name in names:
            module = modules[name]
            weight = module.weight.detach().to(device="cpu", dtype=torch.float32)
            qweight = handle.get_tensor(f"{name}.qweight")
            scales = handle.get_tensor(f"{name}.scales")
            zeros = handle.get_tensor(f"{name}.zeros")
            group_size = weight.shape[1] // scales.shape[1]
            formats = {}
            for packing in PackingFormat:
                decoded = dequantize_groupwise(
                    qweight,
                    scales,
                    zeros=zeros,
                    out_features=weight.shape[0],
                    in_features=weight.shape[1],
                    group_size=group_size,
                    packing=packing,
                )
                formats[packing.value] = tensor_metrics(weight, decoded).to_dict()
            results[name] = formats
    return results


def replace_hf_linear_weights_with_dequantized(
    hf_model,
    checkpoint: str,
    packing: PackingFormat,
) -> None:
    modules = dict(hf_model.named_modules())
    parameters = dict(hf_model.named_parameters())
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        qweight_keys = [key for key in handle.keys() if key.endswith(".qweight")]
        for qweight_key in qweight_keys:
            name = qweight_key.removesuffix(".qweight")
            module = modules[name]
            weight = module.weight
            scales = handle.get_tensor(f"{name}.scales")
            zeros = handle.get_tensor(f"{name}.zeros")
            restored = dequantize_groupwise(
                handle.get_tensor(qweight_key),
                scales,
                zeros=zeros,
                out_features=weight.shape[0],
                in_features=weight.shape[1],
                group_size=weight.shape[1] // scales.shape[1],
                packing=packing,
            )
            weight.data.copy_(restored.to(weight.dtype))
        transformed_keys = [
            key for key in handle.keys() if key.endswith(AWQ_TRANSFORMED_SUFFIXES)
        ]
        for key in transformed_keys:
            if key not in parameters:
                raise KeyError(f"transformed checkpoint parameter is missing from HF model: {key}")
            parameter = parameters[key]
            parameter.data.copy_(handle.get_tensor(key).to(parameter.dtype))


def print_weight_results(results: dict) -> None:
    print("\nWeight-format validation (reference = HF BF16)")
    print(f"{'module':54} {'format':18} {'MAE':>10} {'MAX':>10} {'COS':>11}")
    for name, formats in results.items():
        for packing, metrics in formats.items():
            print(
                f"{name:54} {packing:18} {metrics['mean_abs']:10.6f} "
                f"{metrics['max_abs']:10.6f} {metrics['cosine']:11.8f}"
            )
    best_cosine = max(
        metrics["cosine"]
        for formats in results.values()
        for metrics in formats.values()
    )
    if best_cosine < 0.9:
        print(
            f"INCOMPATIBLE: best sampled weight cosine is {best_cosine:.6f}; "
            "neither packing rule explains this checkpoint layout/source."
        )


def print_activation_results(results: dict, max_rows: int | None) -> None:
    print("\nActivation validation (reference = HF BF16)")
    print(f"{'module':54} {'MAE':>10} {'MAX':>10} {'RMSE':>10} {'COS':>11}")
    items = list(results.items())
    if max_rows is not None:
        items = items[:max_rows]
    for name, metrics in items:
        print(
            f"{name:54} {metrics['mean_abs']:10.6f} {metrics['max_abs']:10.6f} "
            f"{metrics['rmse']:10.6f} {metrics['cosine']:11.8f}"
        )


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
    parser.add_argument("--config", default=str(root / "configs/qwen2-1.5b"))
    parser.add_argument("--backend", choices=("torch", "lpu"), default="torch")
    parser.add_argument(
        "--packing",
        choices=("auto", "canonical", "legacy_signed_add"),
        default="auto",
    )
    parser.add_argument("--device", default="cpu")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-ids", type=parse_ids)
    inputs.add_argument("--prompt")
    parser.add_argument("--output-json", default=str(root / "results/hf_validation.json"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-table-rows", type=int)
    parser.add_argument("--check-architecture-parity", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output_json)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --overwrite explicitly"
        )
    packing = packing_from_checkpoint(args.checkpoint) if args.packing == "auto" else PackingFormat(args.packing)
    group_size = group_size_from_checkpoint(args.checkpoint)

    tokenizer = None
    if args.prompt is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model,
            revision=args.tokenizer_revision,
            local_files_only=args.local_files_only,
        )
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    else:
        input_ids = torch.tensor([args.input_ids], dtype=torch.long)

    print(f"Loading HF reference: {args.hf_model}@{args.hf_revision}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(args.device).eval()
    names = qwen_module_names(hf_model.config.num_hidden_layers)
    weight_results = weight_format_comparison(hf_model, args.checkpoint, DEFAULT_WEIGHT_LAYERS)
    print_weight_results(weight_results)

    with ActivationRecorder(hf_model, set(names)) as recorder, torch.inference_mode():
        hf_logits = hf_model(input_ids.to(args.device), use_cache=False).logits.detach().cpu()
    hf_activations = recorder.activations
    skeleton = QwenInt4ForCausalLM(AutoConfig.from_pretrained(args.config, local_files_only=True))
    non_quantized_state = extract_non_quantized_state(hf_model, skeleton)
    del skeleton
    hf_dequantized_activations = None
    if args.check_architecture_parity:
        print(f"\nRunning HF architecture with dequantized {packing.value} weights")
        replace_hf_linear_weights_with_dequantized(hf_model, args.checkpoint, packing)
        with ActivationRecorder(hf_model, set(names)) as parity_recorder, torch.inference_mode():
            hf_model(input_ids.to(args.device), use_cache=False)
        hf_dequantized_activations = parity_recorder.activations
    del hf_model
    gc.collect()

    print(f"\nLoading quantized model: backend={args.backend}, packing={packing.value}")
    config = AutoConfig.from_pretrained(args.config, local_files_only=True)
    quant_model = QwenInt4ForCausalLM(
        config, backend=args.backend, packing=packing, group_size=group_size
    )
    load_non_quantized_state(quant_model, non_quantized_state)
    load_quantized_checkpoint(quant_model, args.checkpoint)
    quant_model = quant_model.to(args.device).eval()
    with ActivationRecorder(quant_model, set(names)) as recorder, torch.inference_mode():
        quant_logits = quant_model(input_ids.to(args.device)).detach().cpu()
    quant_activations = recorder.activations

    activation_results = {}
    for name in names:
        if name not in hf_activations or name not in quant_activations:
            raise RuntimeError(f"missing captured activation for {name}")
        activation_results[name] = tensor_metrics(
            hf_activations[name], quant_activations[name]
        ).to_dict()
    # Use direct returned logits as a guard against hook/API differences.
    activation_results["returned_logits"] = tensor_metrics(hf_logits, quant_logits).to_dict()
    print_activation_results(activation_results, args.max_table_rows)

    architecture_results = None
    if hf_dequantized_activations is not None:
        architecture_results = {
            name: tensor_metrics(hf_dequantized_activations[name], quant_activations[name]).to_dict()
            for name in names
        }
        print("\nArchitecture/backend parity (reference = HF with identical dequantized weights)")
        print_activation_results(architecture_results, args.max_table_rows)

    first_material = next(
        (
            name
            for name, metrics in activation_results.items()
            if metrics["cosine"] < 0.999 or metrics["mean_abs"] > 0.01
        ),
        None,
    )
    print(f"\nfirst material divergence: {first_material or 'none under configured threshold'}")
    output = {
        "hf_model": args.hf_model,
        "hf_model_revision": args.hf_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "input_ids": input_ids.tolist(),
        "backend": args.backend,
        "packing": packing.value,
        "group_size": group_size,
        "weight_formats": weight_results,
        "activations": activation_results,
        "architecture_parity": architecture_results,
        "first_material_divergence": first_material,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
