"""Create the pinned W4G128 AWQ reference checkpoint on NVIDIA Linux."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_CONFIG = "configs/awq-reference-v1.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    reference = config["reference"]
    calibration = config["calibration"]
    required = {
        "scheme": "W4A16_ASYM",
        "group_size": 128,
        "symmetric": False,
        "native_format": "compressed-tensors",
        "quantization_format": "pack_quantized",
        "pipeline": "sequential",
    }
    mismatches = {
        key: (reference.get(key), value)
        for key, value in required.items()
        if reference.get(key) != value
    }
    if mismatches:
        raise ValueError(f"reference config violates the frozen contract: {mismatches}")
    if calibration["split"] == config["evaluation"]["dataset_split"]:
        raise ValueError("calibration and evaluation splits must be different")
    if calibration["samples"] < 128:
        raise ValueError("the reference run requires at least 128 calibration samples")
    return config


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / EXPECTED_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the frozen plan without importing GPU packages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output_dir = Path(args.output_dir or root / config["output_directory"]).resolve()
    plan = {
        "config": str(config_path),
        "output_directory": str(output_dir),
        "model": config["model"],
        "reference": config["reference"],
        "calibration": config["calibration"],
    }
    if args.validate_only:
        print(json.dumps(plan, indent=2))
        return

    if platform.system() != "Linux":
        raise RuntimeError("the AWQ reference run is restricted to NVIDIA Linux")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AWQ reference run")
    if output_dir.exists():
        allowed_placeholders = {"reference-plan.json"}
        unexpected = sorted(
            path.name for path in output_dir.iterdir() if path.name not in allowed_placeholders
        )
        if unexpected:
            raise FileExistsError(
                f"refusing to overwrite populated artifact directory {output_dir}: "
                f"{unexpected}"
            )

    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = int(config["calibration"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = config["model"]
    calibration_config = config["calibration"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["tokenizer_revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = load_dataset(
        calibration_config["dataset"],
        calibration_config["config"],
        split=calibration_config["split"],
        revision=calibration_config["revision"],
    )
    text_column = calibration_config["text_column"]
    dataset = dataset.filter(lambda row: bool(row[text_column].strip()))
    dataset = dataset.shuffle(seed=seed).select(range(calibration_config["samples"]))

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            row[text_column],
            padding=False,
            max_length=calibration_config["max_sequence_length"],
            truncation=True,
            add_special_tokens=calibration_config["add_special_tokens"],
        )

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    recipe = [
        AWQModifier(
            duo_scaling=config["reference"]["duo_scaling"],
            n_grid=config["reference"]["scale_search_grid_points"],
        ),
        QuantizationModifier(
            targets=config["reference"]["targets"],
            scheme=config["reference"]["scheme"],
            ignore=config["reference"]["ignore"],
        ),
    ]

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=calibration_config["max_sequence_length"],
        num_calibration_samples=calibration_config["samples"],
        pipeline=config["reference"]["pipeline"],
        shuffle_calibration_samples=False,
    )
    elapsed_seconds = time.perf_counter() - started
    peak_gpu_bytes = torch.cuda.max_memory_allocated()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    weight_files = sorted(output_dir.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError("the reference export did not produce a safetensors checkpoint")

    manifest = {
        "schema_version": 1,
        "status": "quantized_pending_quality_validation",
        "config": config,
        "config_sha256": sha256(config_path),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "accelerate",
                "compressed-tensors",
                "llmcompressor",
            )
        },
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_bytes": peak_gpu_bytes,
        "files": {path.name: sha256(path) for path in weight_files},
    }
    (output_dir / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"reference checkpoint: {output_dir}")
    print(f"elapsed seconds: {elapsed_seconds:.2f}")
    print(f"peak GPU bytes: {peak_gpu_bytes}")


if __name__ == "__main__":
    main()
