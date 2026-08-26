"""Phase C evaluation-plan validation and report helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_phase_c_config(
    config_path: str | Path, root: str | Path
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported Phase C evaluation schema")

    scoring = config["scoring"]
    if scoring["accumulation_dtype"] != "float64":
        raise ValueError("Phase C requires float64 NLL accumulation")
    if not 1 <= scoring["stride"] < scoring["window_size"]:
        raise ValueError("scoring stride must be smaller than the window size")
    if config["generation"]["do_sample"] is not False:
        raise ValueError("Phase C generation must use deterministic greedy decoding")

    evaluation_paths: set[Path] = set()
    for corpus in config["corpora"]:
        path = (root_path / corpus["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != corpus["sha256"]:
            raise ValueError(f"checksum mismatch for {path}")
        if corpus.get("calibration_allowed") is not False:
            raise ValueError(f"evaluation corpus is not reserved: {path}")
        evaluation_paths.add(path)

    generation = config["generation"]
    generation_path = (root_path / generation["path"]).resolve()
    if sha256_file(generation_path) != generation["sha256"]:
        raise ValueError(f"checksum mismatch for {generation_path}")
    evaluation_paths.add(generation_path)

    calibration_paths = {
        (root_path / item["path"]).resolve()
        for item in config["calibration_inputs"]
        if "path" in item
    }
    overlap = evaluation_paths & calibration_paths
    if overlap:
        raise ValueError(f"calibration/evaluation path overlap: {sorted(overlap)}")
    calibration_splits = {
        (item.get("dataset"), item.get("revision"), item.get("split"))
        for item in config["calibration_inputs"]
        if "dataset" in item
    }
    for corpus in config["corpora"]:
        source_key = (
            "Salesforce/wikitext" if corpus["id"] == "wikitext2_test_full" else None,
            config["calibration_inputs"][0].get("revision")
            if corpus["id"] == "wikitext2_test_full"
            else None,
            corpus["source_split"],
        )
        if source_key in calibration_splits:
            raise ValueError(f"calibration/evaluation dataset split overlap: {source_key}")
    return config


def model_availability(config: dict[str, Any], root: str | Path) -> dict[str, str]:
    root_path = Path(root).resolve()
    statuses: dict[str, str] = {}
    for model in config["models"]:
        status = model["status"]
        checkpoint_value = model.get("checkpoint")
        if checkpoint_value:
            checkpoint = (root_path / checkpoint_value).resolve()
            if model["kind"] == "compressed_tensors":
                ready = (checkpoint / "runtime-manifest.json").is_file() and bool(
                    list(checkpoint.glob("*.safetensors"))
                )
            else:
                ready = checkpoint.is_file()
            if status == "ready" and not ready:
                raise FileNotFoundError(f"ready model checkpoint is missing: {checkpoint}")
            if status != "ready" and ready:
                status = "artifact_present_requires_status_update"
        statuses[model["id"]] = status
    return statuses


def all_finite(metrics: dict[str, Any]) -> bool:
    numeric = (value for value in metrics.values() if isinstance(value, (int, float)))
    return all(math.isfinite(value) for value in numeric)


def generation_regression(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    compared = min(len(reference), len(candidate))
    matches = sum(left == right for left, right in zip(reference, candidate))
    prefix = 0
    for left, right in zip(reference, candidate):
        if left != right:
            break
        prefix += 1
    unique_ratio = len(set(candidate)) / len(candidate) if candidate else 0.0
    max_token_fraction = (
        max(candidate.count(token) for token in set(candidate)) / len(candidate)
        if candidate
        else 1.0
    )
    return {
        "compared_tokens": compared,
        "position_agreement": matches / compared if compared else 0.0,
        "common_prefix_tokens": prefix,
        "unique_token_ratio": unique_ratio,
        "max_token_fraction": max_token_fraction,
        "empty_continuation": not candidate,
    }
