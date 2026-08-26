import pytest
import torch

from qwen_int4.quantization import (
    PackingFormat,
    dequantize_groupwise,
    pack_int4,
    quantize_activation_weighted_clip,
    quantize_symmetric_rtn,
    unpack_int4,
)


@pytest.mark.parametrize("packing", list(PackingFormat))
def test_pack_unpack_round_trip(packing):
    codes = torch.tensor([-7, -6, -1, 0, 1, 6, 7, -3], dtype=torch.int8)
    packed = pack_int4(codes, packing=packing)
    torch.testing.assert_close(unpack_int4(packed, packing=packing), codes)


def test_canonical_raw_byte_layout():
    # K-even -1 is low nibble 0xF; K-odd 2 is high nibble 0x2.
    packed = pack_int4(torch.tensor([-1, 2], dtype=torch.int8), packing=PackingFormat.CANONICAL)
    assert packed.item() == 0x2F


def test_legacy_borrow_is_not_canonical_unpacking():
    codes = torch.tensor([3, -2], dtype=torch.int8)
    packed = pack_int4(codes, packing=PackingFormat.LEGACY_SIGNED_ADD)
    assert unpack_int4(packed, packing=PackingFormat.CANONICAL).tolist() == [-2, 2]
    assert unpack_int4(packed, packing=PackingFormat.LEGACY_SIGNED_ADD).tolist() == [3, -2]


def test_groupwise_dequantization_indexes_scales_along_k():
    codes = torch.tensor([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=torch.int8)
    scales = torch.tensor([[0.5, 2.0], [1.0, 0.25]], dtype=torch.float16)
    packed = pack_int4(codes, packing=PackingFormat.CANONICAL)
    actual = dequantize_groupwise(
        packed,
        scales,
        out_features=2,
        in_features=4,
        group_size=2,
        packing=PackingFormat.CANONICAL,
        zeros=torch.zeros(2, dtype=torch.uint8),
    )
    expected = torch.tensor([[0.5, 1.0, 6.0, 8.0], [-1.0, -2.0, -0.75, -1.0]])
    torch.testing.assert_close(actual, expected)


def test_nonzero_zero_points_fail_loudly():
    with pytest.raises(NotImplementedError):
        dequantize_groupwise(
            torch.zeros(2, dtype=torch.uint8),
            torch.ones(1, 1),
            out_features=1,
            in_features=4,
            group_size=4,
            zeros=torch.ones(1, dtype=torch.uint8),
        )


def test_symmetric_rtn_quantization_is_self_consistent():
    weight = torch.tensor(
        [[-1.0, -0.75, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]], dtype=torch.float32
    )
    qweight, scales = quantize_symmetric_rtn(weight, group_size=4)
    restored = dequantize_groupwise(
        qweight,
        scales,
        out_features=1,
        in_features=8,
        group_size=4,
        packing=PackingFormat.CANONICAL,
        zeros=torch.zeros(1, dtype=torch.uint8),
    )
    assert qweight.dtype == torch.uint8
    assert scales.shape == (1, 2)
    assert torch.nn.functional.cosine_similarity(weight, restored).item() > 0.99


def test_activation_weighted_clip_returns_groupwise_choice():
    weight = torch.tensor([[10.0, 1.0, 1.0, 1.0, -10.0, -1.0, -1.0, -1.0]])
    importance = torch.tensor([0.001, 1.0, 1.0, 1.0, 0.001, 1.0, 1.0, 1.0])
    qweight, scales, ratios = quantize_activation_weighted_clip(
        weight,
        importance,
        group_size=4,
        clip_ratios=(1.0, 0.5),
    )
    assert qweight.shape == (4, 1)
    assert scales.shape == (1, 2)
    assert ratios.shape == (1, 2)
    assert torch.all(ratios <= 1)
