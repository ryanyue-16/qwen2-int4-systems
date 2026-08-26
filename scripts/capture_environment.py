"""Capture a reproducibility manifest for a versioned INT4 artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open

from qwen_int4.provenance import (
    HF_MODEL_ID,
    HF_MODEL_REVISION,
    HF_TOKENIZER_ID,
    HF_TOKENIZER_REVISION,
    RTN_V4_CHECKPOINT_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "artifacts/qwen2-1.5b-w4g64-rtn-v4/model.safetensors"
DEFAULT_OUTPUT = DEFAULT_CHECKPOINT.with_name("manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_named_path(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or not role or not raw_path:
        raise argparse.ArgumentTypeError("expected ROLE=PATH")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return role, path


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": relative_or_absolute(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_software() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": {
            name: package_version(name)
            for name in ("numpy", "safetensors", "torch", "transformers", "pytest")
        },
    }


def capture_hardware() -> dict[str, object]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    return {
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": cuda_devices,
        },
        "mps": {
            "built": torch.backends.mps.is_built(),
            "available": torch.backends.mps.is_available(),
        },
    }


def checkpoint_record(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, "
            f"found {record['sha256']}"
        )

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
        zero_keys = [key for key in keys if key.endswith(".zeros")]
        nonzero_zero_tensors = [
            key for key in zero_keys if torch.count_nonzero(handle.get_tensor(key)).item()
        ]

    tensor_inventory = {
        "total": len(keys),
        "qweight": sum(key.endswith(".qweight") for key in keys),
        "scales": sum(key.endswith(".scales") for key in keys),
        "zeros": len(zero_keys),
        "nonzero_zero_tensors": nonzero_zero_tensors,
    }
    record.update(
        {
            "expected_sha256": expected_sha256,
            "safetensors_metadata": metadata,
            "tensor_inventory": tensor_inventory,
        }
    )
    return record, metadata


def source_control_record() -> dict[str, object]:
    git_entry = ROOT / ".git"
    if not git_entry.exists():
        return {
            "type": None,
            "available": False,
            "reason": "project root is not a Git work tree",
        }
    return {
        "type": "git",
        "available": True,
        "reason": "Git entry exists; commit capture is not implemented by this script",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=RTN_V4_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="ROLE=PATH",
        help="calibration/evaluation input to checksum; repeatable",
    )
    parser.add_argument(
        "--related",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="ROLE=PATH",
        help="report or config related to the artifact; repeatable",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite explicitly")

    artifact, quantization_contract = checkpoint_record(
        checkpoint, args.expected_checkpoint_sha256
    )
    inputs = {role: file_record(path) for role, path in args.input}
    related = {role: file_record(path) for role, path in args.related}
    if len(inputs) != len(args.input):
        raise ValueError("duplicate --input role")
    if len(related) != len(args.related):
        raise ValueError("duplicate --related role")
    calibration_paths = {
        path.resolve() for role, path in args.input if "calibration" in role
    }
    evaluation_paths = {
        path.resolve() for role, path in args.input if "calibration" not in role
    }

    manifest = {
        "schema_version": "qwen-int4-artifact-manifest-v1",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "artifact": artifact,
        "upstream": {
            "model_id": HF_MODEL_ID,
            "model_revision": HF_MODEL_REVISION,
            "tokenizer_id": HF_TOKENIZER_ID,
            "tokenizer_revision": HF_TOKENIZER_REVISION,
            "revision_status": "retrospectively pinned to the locally validated snapshot",
        },
        "quantization_contract": quantization_contract,
        "random_seeds": {
            "python": args.seed,
            "numpy": args.seed,
            "torch": args.seed,
        },
        "inputs": inputs,
        "data_separation": {
            "calibration_and_evaluation_use_distinct_paths": calibration_paths.isdisjoint(
                evaluation_paths
            ),
            "semantic_leakage_audit": "pending",
        },
        "related_files": related,
        "environment": {
            "software": capture_software(),
            "hardware": capture_hardware(),
        },
        "source_control": source_control_record(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest: {output}")
    print(f"checkpoint sha256: {artifact['sha256']}")
    print(f"input files: {len(inputs)}; related files: {len(related)}")


if __name__ == "__main__":
    main()
