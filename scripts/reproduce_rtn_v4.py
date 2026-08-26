"""Verify the frozen RTN v4 baseline and optionally rerun its quality gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from qwen_int4.provenance import (
    HF_MODEL_REVISION,
    HF_TOKENIZER_REVISION,
    RTN_V4_CHECKPOINT_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/qwen2-1.5b-w4g64-rtn-v4/manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_record_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_file_record(label: str, record: dict) -> None:
    path = resolve_record_path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing {path}")
    actual_size = path.stat().st_size
    if actual_size != record["size_bytes"]:
        raise ValueError(
            f"{label}: size mismatch for {path}: expected {record['size_bytes']}, "
            f"found {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != record["sha256"]:
        raise ValueError(
            f"{label}: SHA-256 mismatch for {path}: expected {record['sha256']}, "
            f"found {actual_sha256}"
        )


def verify_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qwen-int4-artifact-manifest-v1":
        raise ValueError("unsupported manifest schema")
    upstream = manifest["upstream"]
    if upstream["model_revision"] != HF_MODEL_REVISION:
        raise ValueError("manifest model revision differs from the pinned code revision")
    if upstream["tokenizer_revision"] != HF_TOKENIZER_REVISION:
        raise ValueError("manifest tokenizer revision differs from the pinned code revision")
    artifact = manifest["artifact"]
    if artifact["sha256"] != RTN_V4_CHECKPOINT_SHA256:
        raise ValueError("manifest checkpoint SHA-256 differs from the frozen RTN v4 value")
    verify_file_record("artifact", artifact)
    for collection in ("inputs", "related_files"):
        for role, record in manifest[collection].items():
            verify_file_record(f"{collection}.{role}", record)
    if not manifest["data_separation"][
        "calibration_and_evaluation_use_distinct_paths"
    ]:
        raise ValueError("calibration and evaluation paths are not isolated")
    return manifest


def verify_software(manifest: dict) -> None:
    expected_python = manifest["environment"]["software"]["python"]
    actual_python = ".".join(str(item) for item in sys.version_info[:3])
    if actual_python != expected_python:
        raise RuntimeError(
            f"Python version mismatch: expected {expected_python}, found {actual_python}"
        )
    for distribution, expected in manifest["environment"]["software"]["packages"].items():
        if expected is None:
            continue
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required package is missing: {distribution}") from error
        if actual != expected:
            raise RuntimeError(
                f"package mismatch for {distribution}: expected {expected}, found {actual}"
            )


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also rerun versioned HF parity, WikiText-2 2K, and generation reports",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = verify_manifest(manifest_path)
    verify_software(manifest)
    print("manifest, checksums, data paths, and software versions: verified")

    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        source_path
        if not env.get("PYTHONPATH")
        else source_path + os.pathsep + env["PYTHONPATH"]
    )
    checkpoint = resolve_record_path(manifest["artifact"]["path"])
    run(
        [sys.executable, "scripts/inspect_checkpoint.py", str(checkpoint)], env
    )
    run([sys.executable, "-m", "pytest"], env)

    if not args.full:
        print("RTN v4 baseline integrity gate: passed")
        print("Use --full to create new, versioned quality and generation reports.")
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_prefix = ROOT / "results" / f"reproduced_rtn_v4_{run_id}"
    common_revision_args = [
        "--hf-revision",
        HF_MODEL_REVISION,
        "--tokenizer-revision",
        HF_TOKENIZER_REVISION,
        "--local-files-only",
    ]
    run(
        [
            sys.executable,
            "scripts/validate_against_hf.py",
            "--checkpoint",
            str(checkpoint),
            "--input-ids",
            "1,2,3",
            "--backend",
            "torch",
            "--packing",
            "auto",
            "--check-architecture-parity",
            "--output-json",
            str(output_prefix) + "_hf_validation.json",
            "--device",
            args.device,
            *common_revision_args,
        ],
        env,
    )
    run(
        [
            sys.executable,
            "scripts/evaluate_quality.py",
            "--checkpoint",
            str(checkpoint),
            "--text",
            str(ROOT / "data/wikitext2_test_100rows.txt"),
            "--max-tokens",
            "2048",
            "--window-size",
            "256",
            "--stride",
            "128",
            "--output-json",
            str(output_prefix) + "_wikitext2_2k.json",
            "--device",
            args.device,
            *common_revision_args,
        ],
        env,
    )
    run(
        [
            sys.executable,
            "scripts/evaluate_generation.py",
            "--checkpoint",
            str(checkpoint),
            "--max-new-tokens",
            "8",
            "--output-json",
            str(output_prefix) + "_generation.json",
            "--device",
            args.device,
            *common_revision_args,
        ],
        env,
    )
    print(f"full reproduction reports use prefix: {output_prefix}")


if __name__ == "__main__":
    main()
