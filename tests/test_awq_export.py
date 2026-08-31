from pathlib import Path

import pytest
import torch
from torch import nn

from qwen_int4.checkpoint import load_quantized_checkpoint
from qwen_int4.export import (
    AWQ_FORMAT_VERSION,
    build_canonical_awq_export,
    canonical_awq_metadata,
    save_canonical_awq_export,
)
from qwen_int4.linear import QuantLinear
from qwen_int4.quantization import PackingFormat
from scripts.validate_against_hf import replace_hf_linear_weights_with_dequantized


class ToyDenseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(128, elementwise_affine=True, bias=False)
        self.proj = nn.Linear(128, 4, bias=False)


class ToyQuantModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(128, elementwise_affine=True, bias=False)
        self.proj = QuantLinear(
            128, 4, group_size=128, packing=PackingFormat.CANONICAL
        )


def test_canonical_awq_export_has_versioned_asymmetric_contract(tmp_path: Path):
    torch.manual_seed(9)
    model = ToyDenseModel()
    model.input_layernorm.weight.data.fill_(0.75)
    export = build_canonical_awq_export(model)

    assert export.metadata["format_version"] == AWQ_FORMAT_VERSION
    assert export.metadata["group_size"] == "128"
    assert export.metadata["zero_point"] == "canonical_signed_packed_per_group"
    assert export.report["status"] == "awaiting_gpu_reference"
    assert set(export.tensors) == {
        "input_layernorm.weight",
        "proj.qweight",
        "proj.scales",
        "proj.zeros",
    }

    output = tmp_path / "toy-awq-v1"
    checkpoint, report = save_canonical_awq_export(export, output)
    assert checkpoint.is_file()
    assert report.is_file()
    with pytest.raises(FileExistsError):
        save_canonical_awq_export(export, output)


def test_awq_loader_restores_transformed_state_and_asymmetric_weights(tmp_path: Path):
    torch.manual_seed(10)
    source = ToyDenseModel()
    source.input_layernorm.weight.data.fill_(0.625)
    export = build_canonical_awq_export(source)
    checkpoint, _ = save_canonical_awq_export(export, tmp_path / "toy-awq-v1")
    target = ToyQuantModel()

    report = load_quantized_checkpoint(target, checkpoint)

    assert report.loaded_keys == 4
    torch.testing.assert_close(
        target.input_layernorm.weight,
        torch.full_like(target.input_layernorm.weight, 0.625),
    )
    assert torch.count_nonzero(target.proj.zeros).item() > 0


def test_canonical_awq_export_supports_w4g64_contract():
    torch.manual_seed(11)
    model = ToyDenseModel()
    clip_max = model.proj.weight.detach().abs().reshape(4, 2, 64).amax(dim=-1)

    export = build_canonical_awq_export(
        model,
        group_size=64,
        clip_max_by_layer={"proj": clip_max},
    )

    assert export.metadata["group_size"] == "64"
    assert export.tensors["proj.scales"].shape == (4, 2)


def test_canonical_awq_metadata_rejects_unsupported_group_size():
    with pytest.raises(ValueError):
        canonical_awq_metadata(group_size=32)


def test_hf_dequantized_reference_receives_awq_transformed_parameters(tmp_path: Path):
    model = ToyDenseModel()
    transformed = torch.full_like(model.input_layernorm.weight, 0.375)
    checkpoint = tmp_path / "transformed-only.safetensors"
    from safetensors.torch import save_file

    save_file({"input_layernorm.weight": transformed}, str(checkpoint))
    replace_hf_linear_weights_with_dequantized(
        model, str(checkpoint), PackingFormat.CANONICAL
    )

    torch.testing.assert_close(model.input_layernorm.weight, transformed)
