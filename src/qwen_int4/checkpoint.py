"""Strict loading and validation of the local safetensors checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .artifact_contract import (
    CanonicalAWQV6Contract,
    validate_canonical_awq_v6_checkpoint,
)
from .quantization import PackingFormat
from .export import AWQ_FORMAT_VERSION, AWQ_TRANSFORMED_SUFFIXES


@dataclass(frozen=True)
class LoadReport:
    loaded_keys: int
    skipped_non_quantized_keys: int


QUANTIZED_SUFFIXES = (".qweight", ".scales", ".zeros")


def is_quantized_key(name: str) -> bool:
    return name.endswith(QUANTIZED_SUFFIXES)


def checkpoint_metadata(path: str | Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def packing_from_checkpoint(path: str | Path) -> PackingFormat:
    packing = checkpoint_metadata(path).get("packing", "")
    if packing == "canonical_low_nibble_k_even_high_nibble_k_odd":
        return PackingFormat.CANONICAL
    # The unversioned legacy checkpoint has no trustworthy format metadata.
    return PackingFormat.LEGACY_SIGNED_ADD


def group_size_from_checkpoint(path: str | Path, default: int = 128) -> int:
    value = checkpoint_metadata(path).get("group_size")
    return default if value is None else int(value)


def load_quantized_checkpoint(model: torch.nn.Module, path: str | Path) -> LoadReport:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = load_file(str(path), device="cpu")
    expected = model.state_dict()
    expected_quantized = {key for key in expected if is_quantized_key(key)}
    checkpoint_quantized = {key for key in checkpoint if is_quantized_key(key)}
    missing = expected_quantized - checkpoint_quantized
    unexpected = checkpoint_quantized - expected_quantized
    mismatched = {
        key: (tuple(expected[key].shape), tuple(checkpoint[key].shape))
        for key in expected_quantized & checkpoint_quantized
        if expected[key].shape != checkpoint[key].shape
    }
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint contract mismatch:\n"
            f"  missing={sorted(missing)}\n"
            f"  unexpected={sorted(unexpected)}\n"
            f"  shape_mismatches={mismatched}"
        )

    metadata = checkpoint_metadata(path)
    transformed_tensors: dict[str, torch.Tensor] = {}
    if metadata.get("format_version") == AWQ_FORMAT_VERSION:
        expected_transformed = {
            key for key in expected if key.endswith(AWQ_TRANSFORMED_SUFFIXES)
        }
        checkpoint_transformed = {
            key for key in checkpoint if key.endswith(AWQ_TRANSFORMED_SUFFIXES)
        }
        missing_transformed = expected_transformed - checkpoint_transformed
        unexpected_transformed = checkpoint_transformed - expected_transformed
        if missing_transformed or unexpected_transformed:
            raise RuntimeError(
                "AWQ transformed-state contract mismatch:\n"
                f"  missing={sorted(missing_transformed)}\n"
                f"  unexpected={sorted(unexpected_transformed)}"
            )
        transformed_tensors = {
            key: checkpoint[key] for key in checkpoint_transformed
        }

    quantized_tensors = {key: checkpoint[key] for key in checkpoint_quantized}
    load_tensors = {**quantized_tensors, **transformed_tensors}
    incompatible = model.load_state_dict(load_tensors, strict=False)
    unexpected_after_load = set(incompatible.unexpected_keys)
    if unexpected_after_load:
        raise RuntimeError(
            f"quantized state_dict load failed: unexpected={sorted(unexpected_after_load)}"
        )

    nonzero_zero_points = [
        name for name, tensor in model.named_buffers()
        if name.endswith(".zeros") and torch.count_nonzero(tensor).item()
    ]
    zero_point_contract = metadata.get("zero_point", "")
    asymmetric_contracts = {
        "canonical_signed_packed_per_group",
        "asymmetric_canonical_signed_packed_per_group",
    }
    if nonzero_zero_points and zero_point_contract not in asymmetric_contracts:
        raise RuntimeError(
            "non-zero zero-points found without an asymmetric checkpoint contract: "
            + ", ".join(nonzero_zero_points)
        )
    return LoadReport(len(load_tensors), len(checkpoint) - len(load_tensors))


def load_canonical_awq_v6_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    contract: CanonicalAWQV6Contract,
) -> LoadReport:
    """Fail closed on v6 bytes and metadata before loading quantized tensors."""
    validate_canonical_awq_v6_checkpoint(path, contract)
    incompatible_layers = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "qweight")
        and (getattr(module, "group_size", None) != contract.group_size
             or getattr(module, "packing", None) is not PackingFormat.CANONICAL)
    ]
    if incompatible_layers:
        raise RuntimeError(
            "model does not satisfy canonical AWQ v6 W4G64 packing: "
            + ", ".join(incompatible_layers)
        )
    return load_quantized_checkpoint(model, path)


def extract_non_quantized_state(hf_model: torch.nn.Module, target_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy the base-model tensors that the old runtime sourced from Hugging Face.

    Dense HF linear weights are intentionally excluded because QuantLinear owns
    qweight/scales/zeros instead. Embedding, norms, Q/K/V biases and tied lm_head
    are retained.
    """
    source = hf_model.state_dict()
    target = target_model.state_dict()
    result = {
        key: source[key].detach().to("cpu").clone()
        for key in target
        if not is_quantized_key(key) and key in source and source[key].shape == target[key].shape
    }
    required = {
        key
        for key in target
        if not is_quantized_key(key) and (
            key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
            or key.endswith("layernorm.weight")
            or key.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias"))
        )
    }
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"HF base model is missing required non-quantized tensors: {sorted(missing)}")
    return result


def load_non_quantized_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected HF base keys: {sorted(incompatible.unexpected_keys)}")
