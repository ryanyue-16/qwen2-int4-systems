import json
from pathlib import Path

from qwen_int4.evaluation import (
    generation_regression,
    load_and_validate_phase_c_config,
    model_availability,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase_c_plan_has_valid_checksums_and_isolation():
    config = load_and_validate_phase_c_config(
        ROOT / "configs/phase-c-evaluation-v1.json", ROOT
    )

    assert {corpus["id"] for corpus in config["corpora"]} == {
        "wikitext2_test_full",
        "chinese_eval_v1",
    }
    assert config["generation"]["do_sample"] is False
    metadata = json.loads(
        (ROOT / "data/phase_c/wikitext2_test_full.metadata.json").read_text()
    )
    assert metadata["rows"] == 4358
    prompts = json.loads(
        (ROOT / config["generation"]["path"]).read_text(encoding="utf-8")
    )
    assert {item["category"] for item in prompts} == {
        "english",
        "chinese",
        "code",
        "math_logic",
        "long_context",
    }
    assert model_availability(config, ROOT)["w4g128_awq_reference_v1"] == (
        "pending_gpu_validation"
    )


def test_generation_regression_reports_divergence_and_repetition():
    metrics = generation_regression([1, 2, 3, 4], [1, 2, 2, 2])

    assert metrics["common_prefix_tokens"] == 2
    assert metrics["position_agreement"] == 0.5
    assert metrics["max_token_fraction"] == 0.75
    assert metrics["empty_continuation"] is False
