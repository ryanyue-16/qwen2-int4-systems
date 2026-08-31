"""CPU-testable contract checks for the frozen Phase F benchmark protocol."""

import json
from pathlib import Path

import pytest

from scripts.benchmark_phase_f_model import percentile, summarize


ROOT = Path(__file__).resolve().parents[1]


def test_phase_f_benchmark_config_is_frozen_and_bounded():
    config = json.loads(
        (ROOT / "configs/phase-f-model-benchmark-v2.json").read_text(encoding="utf-8")
    )
    assert config["status"] == "correctness_gates_passed_benchmark_authorized"
    assert config["backends"] == ["torch", "triton"]
    assert config["warmup_iterations"] >= 1
    assert config["measurement_iterations"] >= 30
    assert {profile["batch_size"] for profile in config["profiles"]} == {1, 4}
    assert all(profile["new_tokens"] > 0 for profile in config["profiles"])


def test_benchmark_percentiles_use_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.50) == 3.0
    assert percentile(values, 0.95) == pytest.approx(4.8)
    summary = summarize(values)
    assert summary["minimum_ms"] == 1.0
    assert summary["maximum_ms"] == 5.0
    assert summary["mean_ms"] == 3.0
