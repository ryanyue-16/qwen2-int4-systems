"""Correctness-only regression tests for canonical asymmetric W4G64 AWQ."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from qwen_int4.artifact_contract import (
    load_canonical_awq_v6_contract,
    sha256_file,
    validate_canonical_awq_v6_checkpoint,
    validate_canonical_awq_v6_evidence,
)
from qwen_int4.backends import torch_reference_linear
from qwen_int4.checkpoint import load_canonical_awq_v6_checkpoint, load_quantized_checkpoint
from qwen_int4.export import canonical_awq_metadata
from qwen_int4.linear import QuantLinear
from qwen_int4.quantization import (
    PackingFormat,
    dequantize_groupwise,
    pack_int4,
    quantize_asymmetric_rtn,
    unpack_int4,
)
from qwen_int4.transforms import scale_norm_to_linears


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V6_MANIFEST = REPOSITORY_ROOT / "artifacts/qwen2-1.5b-w4g64-awq-canonical-v6/manifest.json"


def _make_contract_checkpoint(tmp_path: Path, *, metadata: dict[str, str] | None = None) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"placeholder": torch.zeros(1)}, str(checkpoint), metadata=canonical_awq_metadata(group_size=64) if metadata is None else metadata)
    manifest = json.loads(V6_MANIFEST.read_text(encoding="utf-8"))
    manifest["artifact"]["sha256"] = sha256_file(checkpoint)
    manifest["artifact"]["size_bytes"] = checkpoint.stat().st_size
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, manifest_path


def test_frozen_v6_manifest_and_repository_evidence_are_valid():
    contract = load_canonical_awq_v6_contract(V6_MANIFEST)
    validate_canonical_awq_v6_evidence(contract, repository_root=REPOSITORY_ROOT)
    assert contract.group_size == 64


def test_v6_checkpoint_metadata_and_bytes_fail_closed(tmp_path: Path):
    checkpoint, manifest_path = _make_contract_checkpoint(tmp_path)
    contract = load_canonical_awq_v6_contract(manifest_path)
    metadata = validate_canonical_awq_v6_checkpoint(checkpoint, contract)
    assert metadata["zero_point"] == "canonical_signed_packed_per_group"

    invalid_metadata = canonical_awq_metadata(group_size=64)
    invalid_metadata["group_size"] = "128"
    invalid_checkpoint, invalid_manifest = _make_contract_checkpoint(
        tmp_path / "invalid", metadata=invalid_metadata
    )
    with pytest.raises(ValueError, match="metadata"):
        validate_canonical_awq_v6_checkpoint(
            invalid_checkpoint, load_canonical_awq_v6_contract(invalid_manifest)
        )


def test_v6_manifest_rejects_a_non_frozen_selection(tmp_path: Path):
    checkpoint, manifest_path = _make_contract_checkpoint(tmp_path)
    del checkpoint
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"]["selected_variant"] = "v7"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selection"):
        load_canonical_awq_v6_contract(manifest_path)


def test_v6_loader_rejects_a_model_with_the_wrong_group_shape(tmp_path: Path):
    checkpoint, manifest_path = _make_contract_checkpoint(tmp_path)
    wrong_group_model = nn.Sequential(
        QuantLinear(128, 4, group_size=128, packing=PackingFormat.CANONICAL)
    )
    with pytest.raises(RuntimeError, match="W4G64"):
        load_canonical_awq_v6_checkpoint(
            wrong_group_model, checkpoint, load_canonical_awq_v6_contract(manifest_path)
        )


def test_quantized_loader_rejects_a_tensor_shape_mismatch(tmp_path: Path):
    source = nn.Sequential(QuantLinear(128, 4, group_size=64, packing=PackingFormat.CANONICAL))
    tensors = source.state_dict()
    tensors["0.qweight"] = tensors["0.qweight"][:-1]
    checkpoint = tmp_path / "bad-shape.safetensors"
    save_file(tensors, str(checkpoint), metadata=canonical_awq_metadata(group_size=64))
    target = nn.Sequential(QuantLinear(128, 4, group_size=64, packing=PackingFormat.CANONICAL))
    with pytest.raises(RuntimeError, match="shape_mismatches"):
        load_quantized_checkpoint(target, checkpoint)


def test_packed_signed_zero_points_use_the_canonical_nibble_order():
    signed_zeros = torch.tensor([-8, -1, 0, 7], dtype=torch.int8)
    packed = pack_int4(signed_zeros, packing=PackingFormat.CANONICAL)
    assert packed.tolist() == [0xF8, 0x70]
    torch.testing.assert_close(
        unpack_int4(packed, packing=PackingFormat.CANONICAL), signed_zeros
    )


def test_asymmetric_w4g64_dequantization_uses_each_group_zero_and_scale():
    codes = torch.tensor([[[-8] * 64, [7] * 64]], dtype=torch.int8).reshape(1, 128)
    scales = torch.tensor([[0.25, 2.0]], dtype=torch.float16)
    zeros = pack_int4(torch.tensor([-7, 5], dtype=torch.int8), packing=PackingFormat.CANONICAL)
    restored = dequantize_groupwise(
        pack_int4(codes, packing=PackingFormat.CANONICAL), scales,
        out_features=1, in_features=128, group_size=64,
        packing=PackingFormat.CANONICAL, zeros=zeros,
    )
    torch.testing.assert_close(restored[:, :64], torch.full((1, 64), -0.25))
    torch.testing.assert_close(restored[:, 64:], torch.full((1, 64), 4.0))


@pytest.mark.parametrize("out_features", [256, 1536], ids=["qwen2-v-proj", "qwen2-q-proj"])
def test_qwen2_linear_shapes_match_explicit_dequantized_reference(out_features: int):
    torch.manual_seed(out_features)
    in_features = 1536
    weight = torch.randn(out_features, in_features) * 0.02
    qweight, scales, zeros = quantize_asymmetric_rtn(weight, group_size=64)
    layer = QuantLinear(in_features, out_features, group_size=64, packing=PackingFormat.CANONICAL)
    layer.qweight.copy_(qweight)
    layer.scales.copy_(scales)
    layer.zeros.copy_(zeros)
    values = torch.stack((
        torch.zeros(in_features),
        torch.linspace(-80, 80, in_features),
        torch.tensor([1.0, -1.0]).repeat(in_features // 2),
    )).reshape(1, 3, in_features).to(torch.bfloat16)
    expected = torch_reference_linear(
        values, qweight, scales, zeros, None, out_features=out_features,
        in_features=in_features, group_size=64, packing=PackingFormat.CANONICAL,
    )
    torch.testing.assert_close(layer(values), expected, rtol=0, atol=0)


def test_transformed_norm_to_linear_parameters_match_dequantized_operator_reference():
    torch.manual_seed(91)
    channels, out_features = 1536, 256
    norm = nn.LayerNorm(channels, elementwise_affine=True, bias=False)
    dense = nn.Linear(channels, out_features, bias=False)
    scales = torch.linspace(0.5, 1.5, channels)
    scale_norm_to_linears(norm, (dense,), scales)
    qweight, group_scales, zeros = quantize_asymmetric_rtn(dense.weight, group_size=64)
    quantized = QuantLinear(channels, out_features, group_size=64, packing=PackingFormat.CANONICAL)
    quantized.qweight.copy_(qweight)
    quantized.scales.copy_(group_scales)
    quantized.zeros.copy_(zeros)
    values = torch.randn(2, 3, channels, dtype=torch.bfloat16)
    transformed = norm(values.to(torch.float32)).to(torch.bfloat16)
    expected = torch_reference_linear(
        transformed, qweight, group_scales, zeros, None, out_features=out_features,
        in_features=channels, group_size=64, packing=PackingFormat.CANONICAL,
    )
    torch.testing.assert_close(quantized(transformed), expected, rtol=0, atol=0)
