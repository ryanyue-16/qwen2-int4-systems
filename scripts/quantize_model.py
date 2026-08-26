"""Generate a versioned, self-describing symmetric group-wise INT4 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from qwen_int4.provenance import HF_MODEL_ID, HF_MODEL_REVISION, HF_TOKENIZER_REVISION
from qwen_int4.quantization import (
    PackingFormat,
    dequantize_groupwise,
    quantize_activation_weighted_clip,
    quantize_symmetric_rtn,
)
from qwen_int4.validation import tensor_metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default=HF_MODEL_ID)
    parser.add_argument("--hf-revision", default=HF_MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=HF_TOKENIZER_REVISION)
    parser.add_argument(
        "--output",
        default=str(root / "artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors"),
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--method", choices=("rtn", "activation_clip"), default="rtn")
    parser.add_argument("--calibration-text", default=str(root / "data/calibration_prompts.txt"))
    parser.add_argument("--calibration-max-tokens", type=int, default=192)
    parser.add_argument("--clip-ratios", default="1.0,0.95,0.9,0.85,0.8,0.75,0.7")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --overwrite explicitly")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading trusted BF16 source: {args.hf_model}@{args.hf_revision}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to("cpu").eval()

    activation_importance: dict[str, torch.Tensor] = {}
    if args.method == "activation_clip":
        calibration_text = Path(args.calibration_text).read_text(encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model,
            revision=args.tokenizer_revision,
            local_files_only=args.local_files_only,
        )
        calibration_ids = tokenizer(
            calibration_text, return_tensors="pt", add_special_tokens=False
        ).input_ids[:, : args.calibration_max_tokens]
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        handles = []

        def make_hook(name: str):
            def collect(_module, inputs, _output) -> None:
                x = inputs[0].detach().to(device="cpu", dtype=torch.float32)
                reduce_dims = tuple(range(x.ndim - 1))
                value = x.square().sum(dim=reduce_dims)
                sums[name] = sums.get(name, torch.zeros_like(value)) + value
                counts[name] = counts.get(name, 0) + x.numel() // x.shape[-1]

            return collect

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name != "lm_head":
                handles.append(module.register_forward_hook(make_hook(name)))
        print(f"Collecting activation statistics from {calibration_ids.shape[1]} calibration tokens")
        with torch.inference_mode():
            model(calibration_ids, use_cache=False)
        for handle in handles:
            handle.remove()
        activation_importance = {name: sums[name] / counts[name] for name in sums}
        print(f"Collected activation statistics for {len(activation_importance)} linear layers")

    quantized: dict[str, torch.Tensor] = {}
    layer_reports: dict[str, dict] = {}
    linear_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
    ]
    clip_ratios = tuple(float(value) for value in args.clip_ratios.split(","))
    print(
        f"Quantizing {len(linear_modules)} linear layers: method={args.method}, "
        f"symmetric W4G{args.group_size}"
    )
    ratio_summaries = {}
    for index, (name, module) in enumerate(linear_modules, start=1):
        weight = module.weight.detach().to(device="cpu", dtype=torch.float32)
        if args.method == "activation_clip":
            qweight, scales, chosen_ratios = quantize_activation_weighted_clip(
                weight,
                activation_importance[name],
                group_size=args.group_size,
                clip_ratios=clip_ratios,
                packing=PackingFormat.CANONICAL,
            )
            ratio_summaries[name] = {
                "mean": chosen_ratios.to(torch.float32).mean().item(),
                "minimum": chosen_ratios.min().item(),
                "fraction_clipped": (chosen_ratios < 1).to(torch.float32).mean().item(),
            }
        else:
            qweight, scales = quantize_symmetric_rtn(
                weight,
                group_size=args.group_size,
                packing=PackingFormat.CANONICAL,
            )
        groups = weight.shape[1] // args.group_size
        zeros = torch.zeros(weight.shape[0] * groups // 2, 1, dtype=torch.uint8)
        quantized[f"{name}.qweight"] = qweight.contiguous()
        quantized[f"{name}.scales"] = scales.contiguous()
        quantized[f"{name}.zeros"] = zeros

        restored = dequantize_groupwise(
            qweight,
            scales,
            zeros=zeros,
            out_features=weight.shape[0],
            in_features=weight.shape[1],
            group_size=args.group_size,
            packing=PackingFormat.CANONICAL,
        )
        metrics = tensor_metrics(weight, restored)
        layer_reports[name] = metrics.to_dict()
        print(
            f"[{index:3d}/{len(linear_modules)}] {name:50} "
            f"mae={metrics.mean_abs:.6f} cos={metrics.cosine:.8f}"
        )

    metadata = {
        "format_version": "qwen-int4-v1",
        "source_model": args.hf_model,
        "source_model_revision": args.hf_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "quantization": "activation_weighted_clip_rtn" if args.method == "activation_clip" else "symmetric_rtn",
        "bits": "4",
        "activation_dtype": "bf16_or_fp16",
        "group_size": str(args.group_size),
        "code_range": "[-7,7]",
        "packing": "canonical_low_nibble_k_even_high_nibble_k_odd",
        "weight_layout": "row_major_[out_features,in_features]",
        "scale_layout": "[out_features,in_features/group_size]",
        "zero_point": "none_symmetric",
    }
    if args.method == "activation_clip":
        metadata["calibration_text"] = str(Path(args.calibration_text).name)
        metadata["calibration_tokens"] = str(calibration_ids.shape[1])
        metadata["clip_ratios"] = args.clip_ratios
    print(f"Saving {len(quantized)} tensors to {output_path}")
    save_file(quantized, str(output_path), metadata=metadata)

    report_path = output_path.with_name("quantization-report.json")
    summary = {
        "metadata": metadata,
        "num_linear_layers": len(linear_modules),
        "worst_weight_cosine": min(item["cosine"] for item in layer_reports.values()),
        "largest_weight_mae": max(item["mean_abs"] for item in layer_reports.values()),
        "layers": layer_reports,
        "clip_ratio_summary": ratio_summaries,
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report: {report_path}")
    print(f"worst weight cosine: {summary['worst_weight_cosine']:.8f}")
    print(f"largest weight MAE: {summary['largest_weight_mae']:.8f}")


if __name__ == "__main__":
    main()
