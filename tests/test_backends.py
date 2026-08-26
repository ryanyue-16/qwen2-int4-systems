import pytest
import torch

from qwen_int4.linear import QuantLinear
from qwen_int4.quantization import (
    PackingFormat,
    pack_int4,
    quantize_asymmetric_rtn,
)


def test_torch_and_lpu_functional_backends_match():
    layer = QuantLinear(
        8,
        4,
        group_size=4,
        bias=True,
        packing=PackingFormat.LEGACY_SIGNED_ADD,
    )
    codes = torch.tensor(
        [
            [1, -2, 3, -4, 5, -6, 7, -1],
            [-1, 2, -3, 4, -5, 6, -7, 1],
            [0, 1, 0, -1, 2, -2, 3, -3],
            [7, -7, 6, -6, 5, -5, 4, -4],
        ],
        dtype=torch.int8,
    )
    layer.qweight.copy_(pack_int4(codes, packing=layer.packing).view_as(layer.qweight))
    layer.scales.copy_(torch.tensor([[0.5, 0.25], [1.0, 0.5], [0.125, 2.0], [0.75, 0.2]]))
    layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.bfloat16))
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    layer.set_backend("torch")
    reference = layer(x)
    layer.set_backend("lpu")
    actual = layer(x)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_unknown_backend_is_rejected():
    layer = QuantLinear(8, 4, group_size=4)
    with pytest.raises(ValueError):
        layer.set_backend("environment-variable-magic")


def test_torch_and_lpu_backends_match_asymmetric_zero_points():
    torch.manual_seed(7)
    layer = QuantLinear(8, 4, group_size=4, packing=PackingFormat.CANONICAL)
    qweight, scales, zeros = quantize_asymmetric_rtn(
        torch.randn(4, 8), group_size=4
    )
    layer.qweight.copy_(qweight.view_as(layer.qweight))
    layer.scales.copy_(scales)
    layer.zeros.copy_(zeros.view_as(layer.zeros))
    values = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    layer.set_backend("torch")
    reference = layer(values)
    layer.set_backend("lpu")
    actual = layer(values)

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
