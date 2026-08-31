"""CUDA-only correctness gates for the Phase E canonical INT4 unpack primitive."""

import pytest
import torch

from qwen_int4.quantization import (
    PackingFormat,
    dequantize_groupwise,
    pack_int4,
    quantize_asymmetric_rtn,
    unpack_int4,
)
from qwen_int4.backends import torch_reference_linear
from qwen_int4.linear import QuantLinear
from qwen_int4.triton_kernels import (
    dequantize_canonical_w4g64_triton,
    linear_canonical_w4g64_triton,
    triton_available,
    unpack_canonical_signed_int4_triton,
)


pytestmark = pytest.mark.skipif(
    not triton_available(), reason="requires CUDA and Triton for Phase E kernel validation"
)


@pytest.mark.parametrize(
    "codes",
    [
        torch.tensor([-8, -7, -1, 0, 1, 6, 7, -8], dtype=torch.int8),
        torch.tensor([7, -8, 0, -1, 3, -4, -5, 6], dtype=torch.int8),
    ],
)
def test_triton_unpack_matches_canonical_boundary_patterns(codes: torch.Tensor):
    packed = pack_int4(codes, packing=PackingFormat.CANONICAL).cuda()
    actual = unpack_canonical_signed_int4_triton(packed)
    torch.testing.assert_close(actual.cpu(), codes)


def test_triton_unpack_matches_pytorch_for_random_bytes_and_odd_element_count():
    torch.manual_seed(20260831)
    packed = torch.randint(0, 256, (513,), dtype=torch.uint8, device="cuda")
    expected = unpack_int4(packed.cpu(), packing=PackingFormat.CANONICAL, elements=1025)
    actual = unpack_canonical_signed_int4_triton(packed, elements=1025)
    torch.testing.assert_close(actual.cpu(), expected)


def test_triton_unpack_rejects_invalid_tensor_contracts():
    packed = torch.zeros(4, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        unpack_canonical_signed_int4_triton(packed.to(torch.int8))
    with pytest.raises(ValueError, match="elements"):
        unpack_canonical_signed_int4_triton(packed, elements=9)
    with pytest.raises(ValueError, match="contiguous"):
        unpack_canonical_signed_int4_triton(packed.repeat(2)[::2])


@pytest.mark.parametrize("shape", [(3, 64), (5, 128), (256, 1536)])
def test_triton_w4g64_dequant_matches_pytorch_reference(shape: tuple[int, int]):
    torch.manual_seed(shape[0] * 10000 + shape[1])
    weight = torch.randn(shape, dtype=torch.float32) * 0.1
    qweight, scales, zeros = quantize_asymmetric_rtn(weight, group_size=64)
    expected = dequantize_groupwise(
        qweight, scales, zeros=zeros, out_features=shape[0], in_features=shape[1],
        group_size=64, packing=PackingFormat.CANONICAL,
    )
    actual = dequantize_canonical_w4g64_triton(
        qweight.cuda(), scales.cuda(), zeros.cuda(),
        out_features=shape[0], in_features=shape[1],
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_triton_w4g64_dequant_rejects_layout_and_dtype_drift():
    qweight = torch.zeros(64, dtype=torch.uint8, device="cuda")
    scales = torch.ones(2, dtype=torch.float16, device="cuda")
    zeros = torch.zeros(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="divisible"):
        dequantize_canonical_w4g64_triton(
            qweight, scales, zeros, out_features=1, in_features=96
        )
    with pytest.raises(ValueError, match="scales.*dtype"):
        dequantize_canonical_w4g64_triton(
            qweight, scales.float(), zeros, out_features=1, in_features=128
        )
    with pytest.raises(ValueError, match="qweight must contain"):
        dequantize_canonical_w4g64_triton(
            qweight[:-1], scales, zeros, out_features=1, in_features=128
        )


@pytest.mark.parametrize(
    ("rows", "out_features", "in_features", "dtype"),
    [
        (1, 256, 1536, torch.float16),
        (8, 1536, 1536, torch.bfloat16),
        (4, 1536, 8960, torch.float16),
    ],
    ids=["v-proj-fp16", "q-proj-bf16", "mlp-down-fp16"],
)
def test_triton_w4g64_linear_matches_pytorch_reference(
    rows: int, out_features: int, in_features: int, dtype: torch.dtype
):
    torch.manual_seed(rows + out_features + in_features)
    weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.02
    qweight, scales, zeros = quantize_asymmetric_rtn(weight, group_size=64)
    values = torch.randn(rows, in_features, dtype=dtype, device="cuda")
    bias = torch.randn(out_features, dtype=dtype, device="cuda") * 0.01
    expected = torch_reference_linear(
        values, qweight.cuda(), scales.cuda(), zeros.cuda(), bias,
        out_features=out_features, in_features=in_features, group_size=64,
        packing=PackingFormat.CANONICAL,
    )
    actual = linear_canonical_w4g64_triton(
        values, qweight.cuda(), scales.cuda(), zeros.cuda(), bias
    )
    tolerance = (0.02, 0.25) if dtype == torch.bfloat16 else (0.01, 0.0625)
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])


def test_triton_w4g64_linear_rejects_noncanonical_contract():
    values = torch.zeros((1, 64), dtype=torch.float16, device="cuda")
    qweight = torch.zeros(32, dtype=torch.uint8, device="cuda")
    scales = torch.ones(1, dtype=torch.float16, device="cuda")
    zeros = torch.zeros(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="qweight size"):
        linear_canonical_w4g64_triton(values, qweight[:-1], scales, zeros)
    with pytest.raises(ValueError, match="activations"):
        linear_canonical_w4g64_triton(values.unsqueeze(0), qweight, scales, zeros)


@pytest.mark.parametrize(
    "input_kind", ["zero", "alternating", "bounded_large"]
)
def test_triton_w4g64_linear_matches_adversarial_inputs(input_kind: str):
    torch.manual_seed(81)
    out_features, in_features = 64, 128
    qweight, scales, zeros = quantize_asymmetric_rtn(
        torch.randn(out_features, in_features) * 0.05, group_size=64
    )
    if input_kind == "zero":
        values = torch.zeros((2, in_features), dtype=torch.float16, device="cuda")
    elif input_kind == "alternating":
        values = torch.tensor([1.0, -1.0], device="cuda").repeat(2, in_features // 2).to(torch.float16)
    else:
        values = torch.linspace(-8, 8, in_features, device="cuda").repeat(2, 1).to(torch.float16)
    expected = torch_reference_linear(
        values, qweight.cuda(), scales.cuda(), zeros.cuda(), None,
        out_features=out_features, in_features=in_features, group_size=64,
        packing=PackingFormat.CANONICAL,
    )
    actual = linear_canonical_w4g64_triton(values, qweight.cuda(), scales.cuda(), zeros.cuda())
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.0625)


def test_quantlinear_triton_dispatch_matches_torch_reference():
    torch.manual_seed(117)
    layer = QuantLinear(128, 64, group_size=64, bias=True, packing=PackingFormat.CANONICAL).cuda()
    qweight, scales, zeros = quantize_asymmetric_rtn(torch.randn(64, 128) * 0.1, group_size=64)
    layer.qweight.copy_(qweight.cuda())
    layer.scales.copy_(scales.cuda())
    layer.zeros.copy_(zeros.cuda())
    layer.bias.copy_(torch.randn(64, dtype=torch.bfloat16, device="cuda") * 0.01)
    values = torch.randn(2, 3, 128, dtype=torch.bfloat16, device="cuda")
    layer.set_backend("torch")
    expected = layer(values)
    layer.set_backend("triton")
    actual = layer(values)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


def test_quantlinear_triton_dispatch_preserves_fp32_reference_path():
    torch.manual_seed(118)
    layer = QuantLinear(
        128, 64, group_size=64, packing=PackingFormat.CANONICAL
    ).cuda()
    qweight, scales, zeros = quantize_asymmetric_rtn(
        torch.randn(64, 128) * 0.1, group_size=64
    )
    layer.qweight.copy_(qweight.cuda())
    layer.scales.copy_(scales.cuda())
    layer.zeros.copy_(zeros.cuda())
    values = torch.randn(2, 128, dtype=torch.float32, device="cuda")
    layer.set_backend("torch")
    expected = layer(values)
    layer.set_backend("triton")
    actual = layer(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
