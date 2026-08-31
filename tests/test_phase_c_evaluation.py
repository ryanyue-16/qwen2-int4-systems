import json
from pathlib import Path

from qwen_int4.evaluation import (
    generation_regression,
    load_and_validate_phase_c_config,
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


def test_phase_c_v2_selects_the_hashed_v6_candidate():
    config = load_and_validate_phase_c_config(
        ROOT / "configs/phase-c-evaluation-v2.json", ROOT
    )

    canonical = next(
        model
        for model in config["models"]
        if model["id"] == "w4g64_awq_canonical_v6"
    )
    assert canonical["checkpoint"].endswith("v6-gpu-r1/model.safetensors")
    assert canonical["sha256"] == (
        "131a020962efc4e236b17fbd7da02f4aa9914e4c88a07b647e8cfe47dcaecab6"
    )
    assert "w4g64_rtn_v4" not in {model["id"] for model in config["models"]}


def test_phase_c_v3_freezes_the_final_logits_gate_for_v6():
    config = load_and_validate_phase_c_config(
        ROOT / "configs/phase-c-evaluation-v3.json", ROOT
    )

    assert config["evaluation_id"] == "phase-c-evaluation-v3"
    assert config["gates"]["architecture_backend_final_logits_min_cosine"] == 0.9997
    assert config["gates"]["canonical_awq_max_bf16_perplexity_ratio"] == 1.05
    assert "w4g64_rtn_v4" not in {model["id"] for model in config["models"]}


def test_v6_manifest_links_the_final_phase_c_evidence():
    manifest = json.loads(
        (
            ROOT
            / "artifacts/qwen2-1.5b-w4g64-awq-canonical-v6/manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "frozen_phase_d_candidate"
    assert manifest["artifact"]["sha256"] == (
        "131a020962efc4e236b17fbd7da02f4aa9914e4c88a07b647e8cfe47dcaecab6"
    )
    assert manifest["gates"]["final_logits_cosine"] >= manifest["gates"][
        "final_logits_cosine_min"
    ]
    assert manifest["evidence"]["phase_c_final"]["path"] == (
        "results/phase_c_evaluation_v3_gpu_run1.json"
    )


def test_generation_regression_reports_divergence_and_repetition():
    metrics = generation_regression([1, 2, 3, 4], [1, 2, 2, 2])

    assert metrics["common_prefix_tokens"] == 2
    assert metrics["position_agreement"] == 0.5
    assert metrics["max_token_fraction"] == 0.75
    assert metrics["empty_continuation"] is False
