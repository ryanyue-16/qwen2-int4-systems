"""Canonical asymmetric AWQ checkpoint construction and versioned serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import save_file
from torch import nn

from .provenance import HF_MODEL_ID, HF_MODEL_REVISION, HF_TOKENIZER_REVISION
from .quantization import (
    PackingFormat,
    dequantize_groupwise,
    quantize_asymmetric_rtn,
)
from .validation import tensor_metrics


AWQ_FORMAT_VERSION = "qwen-int4-awq-v1"
AWQ_TRANSFORMED_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "v_proj.bias",
)


@dataclass(frozen=True)
class CanonicalAWQExport:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, str]
    report: dict


def canonical_awq_metadata(*, group_size: int = 128) -> dict[str, str]:
    if group_size not in {64, 128}:
        raise ValueError("canonical AWQ supports group sizes 64 and 128")
    return {
        "format_version": AWQ_FORMAT_VERSION,
        "source_model": HF_MODEL_ID,
        "source_model_revision": HF_MODEL_REVISION,
        "tokenizer_revision": HF_TOKENIZER_REVISION,
        "quantization": "canonical_awq_w4a16_asymmetric",
        "bits": "4",
        "activation_dtype": "bf16_or_fp16",
        "group_size": str(group_size),
        "code_range": "[-8,7]",
        "packing": "canonical_low_nibble_k_even_high_nibble_k_odd",
        "weight_layout": "row_major_[out_features,in_features]",
        "scale_layout": "[out_features,in_features/group_size]",
        "zero_point": "canonical_signed_packed_per_group",
        "zero_layout": "packed_row_major_[out_features,in_features/group_size]",
        "status": "awaiting_gpu_reference",
    }


def build_canonical_awq_export(
    model: nn.Module,
    *,
    group_size: int = 128,
    clip_max_by_layer: Mapping[str, torch.Tensor] | None = None,
) -> CanonicalAWQExport:
    """Quantize an already AWQ-transformed model into the canonical contract."""

    metadata = canonical_awq_metadata(group_size=group_size)
    clip_max_by_layer = {} if clip_max_by_layer is None else dict(clip_max_by_layer)
    tensors: dict[str, torch.Tensor] = {}
    layer_reports: dict[str, dict] = {}
    linear_names: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        linear_names.add(name)
        weight = module.weight.detach().to(device="cpu", dtype=torch.float32)
        qweight, scales, zeros = quantize_asymmetric_rtn(
            weight,
            group_size=group_size,
            clip_max=clip_max_by_layer.get(name),
            packing=PackingFormat.CANONICAL,
        )
        tensors[f"{name}.qweight"] = qweight.contiguous()
        tensors[f"{name}.scales"] = scales.contiguous()
        tensors[f"{name}.zeros"] = zeros.contiguous()
        restored = dequantize_groupwise(
            qweight,
            scales,
            out_features=module.out_features,
            in_features=module.in_features,
            group_size=group_size,
            packing=PackingFormat.CANONICAL,
            zeros=zeros,
        )
        layer_reports[name] = tensor_metrics(weight, restored).to_dict()

    unknown_clips = set(clip_max_by_layer) - linear_names
    if unknown_clips:
        raise KeyError(f"clip limits target unknown linear layers: {sorted(unknown_clips)}")
    for name, parameter in model.named_parameters():
        if name.endswith(AWQ_TRANSFORMED_SUFFIXES):
            tensors[name] = parameter.detach().to("cpu").contiguous()
    if not linear_names:
        raise ValueError("model has no quantizable linear layers")
    report = {
        "schema_version": 1,
        "status": "awaiting_gpu_reference",
        "metadata": metadata,
        "num_linear_layers": len(linear_names),
        "num_transformed_parameters": sum(
            name.endswith(AWQ_TRANSFORMED_SUFFIXES) for name in tensors
        ),
        "worst_weight_cosine": min(
            values["cosine"] for values in layer_reports.values()
        ),
        "layers": layer_reports,
    }
    return CanonicalAWQExport(tensors, metadata, report)


def save_canonical_awq_export(
    export: CanonicalAWQExport,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Save a new versioned export without overwriting any existing directory."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    checkpoint = output / "model.safetensors"
    report = output / "quantization-report.json"
    save_file(export.tensors, str(checkpoint), metadata=export.metadata)
    report.write_text(json.dumps(export.report, indent=2) + "\n", encoding="utf-8")
    return checkpoint, report
