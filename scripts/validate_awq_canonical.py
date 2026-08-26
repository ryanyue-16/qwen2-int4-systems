"""Validate the CPU-testable Phase D contract without creating model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_int4.evaluation import sha256_file
from qwen_int4.export import AWQ_FORMAT_VERSION, canonical_awq_metadata


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/awq-canonical-v1.json"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config["quantization"]
    calibration = config["calibration"]
    if config["status"] != "awaiting_gpu_reference":
        raise ValueError("Phase D must remain awaiting_gpu_reference on this host")
    if quantization["group_size"] != 128 or quantization["symmetric"] is not False:
        raise ValueError("Phase D requires asymmetric W4G128")
    metadata = canonical_awq_metadata(group_size=quantization["group_size"])
    if metadata["format_version"] != AWQ_FORMAT_VERSION:
        raise ValueError("canonical AWQ format version drift")
    calibration_path = (root / calibration["path"]).resolve()
    if sha256_file(calibration_path) != calibration["sha256"]:
        raise ValueError("calibration checksum mismatch")

    output = (root / config["output_directory"]).resolve()
    checkpoint = output / "model.safetensors"
    if checkpoint.exists():
        raise RuntimeError(
            "an unvalidated canonical AWQ checkpoint exists; inspect it before changing status"
        )
    report = {
        "schema_version": 1,
        "artifact_id": config["artifact_id"],
        "status": config["status"],
        "config_sha256": sha256_file(config_path),
        "calibration_sha256": calibration["sha256"],
        "format_version": AWQ_FORMAT_VERSION,
        "weights_present": False,
        "cpu_testable_contract": "valid",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
