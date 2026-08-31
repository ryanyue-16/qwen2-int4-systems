"""Fail-closed operator-level autotune for the frozen canonical AWQ v6 kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import torch
from safetensors import safe_open

from qwen_int4.backends import torch_reference_linear
from qwen_int4.quantization import PackingFormat
from qwen_int4.triton_kernels import linear_canonical_w4g64_triton, triton_available


LAYER_BY_PROFILE = {
    "v_projection_fp16": "model.layers.0.self_attn.v_proj",
    "q_projection_bf16": "model.layers.0.self_attn.q_proj",
    "mlp_down_projection_fp16": "model.layers.0.mlp.down_proj",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_candidate(
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    candidate: dict[str, int],
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        linear_canonical_w4g64_triton(activations, qweight, scales, zeros, **candidate)
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        linear_canonical_w4g64_triton(activations, qweight, scales, zeros, **candidate)
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop))
    return timings


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(root / "configs/phase-e-triton-autotune-v1.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint, config_path, output = Path(args.checkpoint), Path(args.config), Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing autotune report: {output}")
    if not triton_available():
        raise RuntimeError("Phase E autotune requires CUDA and Triton")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "correctness_validated_autotune_authorized":
        raise ValueError("autotune config is not authorized by the correctness gate")
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 does not match the frozen v6 config")
    dtype_by_name = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    results = []
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for profile_index, profile in enumerate(config["profiles"]):
            prefix = LAYER_BY_PROFILE[profile["name"]]
            rows, expected_out, expected_in = profile["shape_mnk"]
            qweight = handle.get_tensor(f"{prefix}.qweight").cuda()
            scales = handle.get_tensor(f"{prefix}.scales").cuda()
            zeros = handle.get_tensor(f"{prefix}.zeros").cuda()
            if tuple(scales.shape) != (expected_out, expected_in // config["group_size"]):
                raise ValueError(f"frozen v6 tensor shape drift for {prefix}")
            torch.manual_seed(20260831 + profile_index)
            activations = torch.randn(
                rows, expected_in, dtype=dtype_by_name[profile["dtype"]], device="cuda"
            )
            reference = torch_reference_linear(
                activations, qweight, scales, zeros, None,
                out_features=expected_out, in_features=expected_in,
                group_size=config["group_size"], packing=PackingFormat.CANONICAL,
            )
            candidates = []
            for candidate in config["candidates"]:
                actual = linear_canonical_w4g64_triton(
                    activations, qweight, scales, zeros, **candidate
                )
                torch.testing.assert_close(actual, reference, rtol=profile["rtol"], atol=profile["atol"])
                timings = run_candidate(
                    activations, qweight, scales, zeros, candidate,
                    warmup=config["warmup_iterations"], iterations=config["measurement_iterations"],
                )
                candidates.append({
                    "candidate": candidate,
                    "median_ms": median(timings),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                    "parity": "passed",
                })
            selected = min(candidates, key=lambda value: value["median_ms"])
            results.append({"profile": profile, "layer": prefix, "candidates": candidates, "selected": selected})
    report = {
        "schema_version": 1,
        "phase": "E",
        "status": "correctness_validated_operator_autotune_completed",
        "checkpoint_sha256": config["checkpoint_sha256"],
        "config": str(config_path),
        "selection_rule": config["selection"],
        "results": results,
        "limitations": [
            "Results are CUDA-event operator timings only.",
            "No model, prefill, decode, batching, memory, latency, or throughput claim is made.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
