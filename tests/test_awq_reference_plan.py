import json
from pathlib import Path

from scripts.quantize_awq_reference import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_awq_reference_contract_is_frozen_and_split_safe():
    config = load_config(ROOT / "configs/awq-reference-v1.json")

    assert config["reference"]["scheme"] == "W4A16_ASYM"
    assert config["reference"]["group_size"] == 128
    assert config["reference"]["symmetric"] is False
    assert config["reference"]["quantization_format"] == "pack_quantized"
    assert config["calibration"]["split"] == "train"
    assert config["evaluation"]["dataset_split"] == "test"


def test_awq_reference_plan_records_the_pre_gpu_contract():
    artifact = ROOT / "artifacts/qwen2-1.5b-w4g128-awq-reference-v1"
    plan = json.loads((artifact / "reference-plan.json").read_text(encoding="utf-8"))

    assert plan["status"] == "pending_gpu_validation"
    assert plan["weights_present"] is False
