"""Validate the frozen Phase C plan and write an optional pending-status report."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from qwen_int4.evaluation import (
    load_and_validate_phase_c_config,
    model_availability,
    sha256_file,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(root / "configs/phase-c-evaluation-v1.json")
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_and_validate_phase_c_config(config_path, root)
    statuses = model_availability(config, root)
    report = {
        "schema_version": 1,
        "evaluation_id": config["evaluation_id"],
        "status": "pending_model_runs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "model_status": statuses,
        "corpora": {
            corpus["id"]: {
                "path": corpus["path"],
                "sha256": corpus["sha256"],
            }
            for corpus in config["corpora"]
        },
        "generation": {
            "path": config["generation"]["path"],
            "sha256": config["generation"]["sha256"],
        },
        "calibration_evaluation_isolation": "passed",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "missing_required_runs": [
            model_id for model_id, status in statuses.items() if status != "gpu_validated"
        ],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output_json:
        output = Path(args.output_json).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"report: {output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
