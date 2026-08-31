"""Fail-closed validation for the frozen canonical AWQ v6 artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from safetensors import safe_open


CANONICAL_AWQ_V6_ARTIFACT_ID = "qwen2-1.5b-w4g64-awq-canonical-v6"
CANONICAL_AWQ_V6_FORMAT_VERSION = "qwen-int4-awq-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CanonicalAWQV6Contract:
    manifest_path: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    group_size: int
    model_id: str
    model_revision: str
    tokenizer_revision: str


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a regular file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest field {field!r} must be an object")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {field!r} must be a non-empty string")
    return value


def load_canonical_awq_v6_contract(manifest_path: str | Path) -> CanonicalAWQV6Contract:
    """Load the immutable v6 manifest and reject contract drift."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid manifest JSON: {path}") from error
    root = _require_mapping(manifest, "root")
    if root.get("schema_version") != "qwen-int4-awq-artifact-manifest-v1":
        raise ValueError("unsupported canonical AWQ manifest schema")
    if root.get("artifact_id") != CANONICAL_AWQ_V6_ARTIFACT_ID:
        raise ValueError("manifest is not the canonical AWQ v6 artifact")
    if root.get("status") != "frozen_phase_d_candidate":
        raise ValueError("canonical AWQ v6 manifest must remain frozen")

    artifact = _require_mapping(root.get("artifact"), "artifact")
    checkpoint_sha256 = _require_string(artifact.get("sha256"), "artifact.sha256")
    if not _SHA256_RE.fullmatch(checkpoint_sha256):
        raise ValueError("artifact.sha256 must be a lowercase SHA-256 digest")
    checkpoint_size_bytes = artifact.get("size_bytes")
    if not isinstance(checkpoint_size_bytes, int) or checkpoint_size_bytes <= 0:
        raise ValueError("artifact.size_bytes must be a positive integer")
    remote_path = _require_string(artifact.get("remote_path"), "artifact.remote_path")
    if not remote_path.startswith("/") or not remote_path.endswith("/model.safetensors"):
        raise ValueError("artifact.remote_path must be an absolute model.safetensors path")

    upstream = _require_mapping(root.get("upstream"), "upstream")
    model_id = _require_string(upstream.get("model_id"), "upstream.model_id")
    model_revision = _require_string(upstream.get("model_revision"), "upstream.model_revision")
    tokenizer_revision = _require_string(upstream.get("tokenizer_revision"), "upstream.tokenizer_revision")

    quantization = _require_mapping(root.get("quantization_contract"), "quantization_contract")
    if quantization.get("format_version") != CANONICAL_AWQ_V6_FORMAT_VERSION:
        raise ValueError("canonical AWQ v6 format version mismatch")
    if quantization.get("bits") != 4 or quantization.get("group_size") != 64:
        raise ValueError("canonical AWQ v6 requires W4G64")
    if quantization.get("quantization") != "canonical_awq_asymmetric":
        raise ValueError("canonical AWQ v6 requires asymmetric quantization")
    if quantization.get("zero_point") != "per-group asymmetric":
        raise ValueError("canonical AWQ v6 requires per-group asymmetric zero points")
    if quantization.get("packing") != "canonical signed INT4 packing":
        raise ValueError("canonical AWQ v6 requires canonical signed INT4 packing")

    selection = _require_mapping(root.get("selection"), "selection")
    if selection.get("selected_variant") != "v6" or selection.get("rejected_variant") != "v7":
        raise ValueError("canonical AWQ v6 selection record mismatch")
    if selection.get("phase_e_eligibility") != "correctness-only work; no performance claim is authorized by this manifest":
        raise ValueError("manifest must retain the Phase E correctness-only restriction")

    return CanonicalAWQV6Contract(
        manifest_path=path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=checkpoint_size_bytes,
        group_size=64,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )


def validate_canonical_awq_v6_evidence(
    contract: CanonicalAWQV6Contract, *, repository_root: str | Path
) -> None:
    """Verify every repository-resident evidence checksum named by the manifest."""
    manifest = json.loads(contract.manifest_path.read_text(encoding="utf-8"))
    evidence = _require_mapping(manifest.get("evidence"), "evidence")
    root = Path(repository_root).resolve()
    for name, record_value in evidence.items():
        record = _require_mapping(record_value, f"evidence.{name}")
        relative_path = _require_string(record.get("path"), f"evidence.{name}.path")
        expected_digest = _require_string(record.get("sha256"), f"evidence.{name}.sha256")
        candidate = (root / relative_path).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"evidence.{name} path is missing or escapes the repository")
        if sha256_file(candidate) != expected_digest:
            raise ValueError(f"evidence.{name} checksum mismatch")
        config_path = record.get("config_path")
        if config_path is not None:
            config_digest = _require_string(record.get("config_sha256"), f"evidence.{name}.config_sha256")
            config_candidate = (root / _require_string(config_path, f"evidence.{name}.config_path")).resolve()
            if root not in config_candidate.parents or not config_candidate.is_file():
                raise ValueError(f"evidence.{name} config path is missing or escapes the repository")
            if sha256_file(config_candidate) != config_digest:
                raise ValueError(f"evidence.{name} config checksum mismatch")


def validate_canonical_awq_v6_checkpoint(
    checkpoint_path: str | Path, contract: CanonicalAWQV6Contract
) -> dict[str, str]:
    """Validate checkpoint bytes and safetensors metadata before model loading."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != contract.checkpoint_size_bytes:
        raise ValueError("checkpoint size does not match the frozen v6 manifest")
    if sha256_file(path) != contract.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match the frozen v6 manifest")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected = {
        "format_version": CANONICAL_AWQ_V6_FORMAT_VERSION,
        "source_model": contract.model_id,
        "source_model_revision": contract.model_revision,
        "tokenizer_revision": contract.tokenizer_revision,
        "quantization": "canonical_awq_w4a16_asymmetric",
        "bits": "4",
        "group_size": str(contract.group_size),
        "code_range": "[-8,7]",
        "packing": "canonical_low_nibble_k_even_high_nibble_k_odd",
        "weight_layout": "row_major_[out_features,in_features]",
        "scale_layout": "[out_features,in_features/group_size]",
        "zero_point": "canonical_signed_packed_per_group",
        "zero_layout": "packed_row_major_[out_features,in_features/group_size]",
    }
    mismatch = {key: (value, metadata.get(key)) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise ValueError(f"checkpoint metadata does not satisfy canonical AWQ v6: {mismatch}")
    return metadata
