"""Validate frozen canonical AWQ v6 provenance and an optional local checkpoint copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_int4.artifact_contract import (
    load_canonical_awq_v6_contract,
    validate_canonical_awq_v6_checkpoint,
    validate_canonical_awq_v6_evidence,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(root / "artifacts/qwen2-1.5b-w4g64-awq-canonical-v6/manifest.json"),
    )
    parser.add_argument(
        "--checkpoint",
        help="Local path to the remote v6 model.safetensors copy; omitted validates manifest evidence only.",
    )
    args = parser.parse_args()

    contract = load_canonical_awq_v6_contract(args.manifest)
    validate_canonical_awq_v6_evidence(contract, repository_root=root)
    checkpoint_validated = args.checkpoint is not None
    if checkpoint_validated:
        validate_canonical_awq_v6_checkpoint(args.checkpoint, contract)
    print(json.dumps({
        "artifact_id": "qwen2-1.5b-w4g64-awq-canonical-v6",
        "contract": "valid",
        "evidence": "valid",
        "checkpoint_validated": checkpoint_validated,
    }, indent=2))


if __name__ == "__main__":
    main()
