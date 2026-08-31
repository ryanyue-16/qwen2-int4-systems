import pytest
import torch
from torch import nn

from qwen_int4.backends import lpu_functional_linear, torch_reference_linear
from qwen_int4.linear import QuantLinear
from qwen_int4.quantization import (
    PackingFormat,
    pack_int4,
    quantize_asymmetric_rtn,
)


GROUP_SIZE = 64


def _canonical_tensors(
    *, out_features: int = 6, in_features: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(out_features * in_features, dtype=torch.float32)
    weight = ((values.remainder(31) - 15) / 8).reshape(out_features, in_features)
    return quantize_asymmetric_rtn(weight, group_size=GROUP_SIZE)


def _run_backends(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    arguments = {
        "out_features": scales.shape[0],
        "in_features": x.shape[-1],
        "group_size": GROUP_SIZE,
        "packing": PackingFormat.CANONICAL,
    }
    reference = torch_reference_linear(
        x, qweight, scales, zeros, bias, **arguments
    )
    actual = lpu_functional_linear(
        x, qweight, scales, zeros, bias, **arguments
    )
    return reference, actual


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("with_bias", [False, True])
def test_lpu_functional_matches_torch_reference(dtype, rows, with_bias):
    qweight, scales, zeros = _canonical_tensors()
    base = torch.linspace(-3, 3, rows * 128, dtype=torch.float32)
    x = base.reshape(rows, 128).to(dtype)
    bias = torch.linspace(-0.25, 0.25, 6).to(dtype) if with_bias else None

    reference, actual = _run_backends(x, qweight, scales, zeros, bias)

    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros(2, 128),
        torch.linspace(-200, 200, 256).reshape(2, 128),
        torch.tensor([1.0, -1.0]).repeat(128).reshape(2, 128),
    ],
    ids=["zero", "large-magnitude", "alternating-sign"],
)
def test_lpu_functional_deterministic_activation_edges(values):
    qweight, scales, zeros = _canonical_tensors()
    reference, actual = _run_backends(
        values.to(torch.bfloat16), qweight, scales, zeros
    )

    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-2)


def test_lpu_functional_supports_non_contiguous_activations():
    qweight, scales, zeros = _canonical_tensors()
    x = torch.arange(4 * 128, dtype=torch.float32).reshape(128, 4).t()
    assert not x.is_contiguous()

    reference, actual = _run_backends(
        x.to(torch.bfloat16), qweight, scales, zeros
    )

    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-2)


def test_lpu_functional_handles_signed_int4_boundary_zero_points():
    out_features, in_features = 2, 128
    codes = torch.tensor([-8, 7]).repeat(out_features * in_features // 2)
    signed_zeros = torch.tensor([-8, 7, 7, -8], dtype=torch.int8)
    qweight = pack_int4(codes, packing=PackingFormat.CANONICAL)
    zeros = pack_int4(signed_zeros, packing=PackingFormat.CANONICAL)
    scales = torch.tensor([[0.25, 0.5], [1.0, 2.0]], dtype=torch.float16)
    x = torch.tensor([1.0, -1.0]).repeat(in_features // 2).reshape(1, in_features)

    reference, actual = _run_backends(
        x.to(torch.float16), qweight, scales, zeros
    )

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_lpu_backend_dispatch_supports_awq_transformed_inputs_and_bias():
    dense = nn.Linear(128, 6, bias=True)
    with torch.no_grad():
        dense.weight.copy_(
            torch.linspace(-1, 1, dense.weight.numel()).reshape_as(dense.weight)
        )
        dense.bias.copy_(torch.linspace(-0.2, 0.2, 6))
        transform = torch.linspace(0.5, 1.5, 128)
        dense.weight.mul_(transform)
    qweight, scales, zeros = quantize_asymmetric_rtn(
        dense.weight, group_size=GROUP_SIZE
    )
    layer = QuantLinear(
        128,
        6,
        group_size=GROUP_SIZE,
        bias=True,
        packing=PackingFormat.CANONICAL,
        backend="lpu",
    )
    layer.qweight.copy_(qweight.view_as(layer.qweight))
    layer.scales.copy_(scales)
    layer.zeros.copy_(zeros.view_as(layer.zeros))
    layer.bias.copy_(dense.bias.to(torch.bfloat16))
    x = torch.linspace(-2, 2, 2 * 3 * 128).reshape(2, 3, 128).to(torch.bfloat16)

    actual = layer(x)
    layer.set_backend("torch")
    reference = layer(x)

    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


def test_unknown_backend_is_rejected():
    layer = QuantLinear(8, 4, group_size=4)
    with pytest.raises(ValueError):
        layer.set_backend("environment-variable-magic")


def test_two_layer_lpu_dispatch_smoke_matches_torch_functional_behavior():
    layers = [
        QuantLinear(
            128,
            128,
            group_size=GROUP_SIZE,
            packing=PackingFormat.CANONICAL,
            backend="lpu",
        )
        for _ in range(2)
    ]
    for index, layer in enumerate(layers):
        qweight, scales, zeros = _canonical_tensors(
            out_features=128, in_features=128
        )
        layer.qweight.copy_(qweight.view_as(layer.qweight))
        layer.scales.copy_(scales * (index + 1))
        layer.zeros.copy_(zeros.view_as(layer.zeros))
    values = torch.linspace(-1, 1, 256).reshape(2, 128).to(torch.bfloat16)

    lpu_output = values
    for layer in layers:
        lpu_output = torch.tanh(layer(lpu_output).to(torch.float32)).to(torch.bfloat16)
    for layer in layers:
        layer.set_backend("torch")
    torch_output = values
    for layer in layers:
        torch_output = torch.tanh(layer(torch_output).to(torch.float32)).to(torch.bfloat16)

    torch.testing.assert_close(lpu_output, torch_output, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        ({"group_size": 128}, ValueError, "W4G64"),
        (
            {"packing": PackingFormat.LEGACY_SIGNED_ADD},
            ValueError,
            "canonical INT4 packing",
        ),
        ({"in_features": 64}, ValueError, "activation feature dimension"),
    ],
)
def test_lpu_functional_rejects_incompatible_mapping(override, error, message):
    qweight, scales, zeros = _canonical_tensors()
    arguments = {
        "out_features": 6,
        "in_features": 128,
        "group_size": GROUP_SIZE,
        "packing": PackingFormat.CANONICAL,
    }
    arguments.update(override)

    with pytest.raises(error, match=message):
        lpu_functional_linear(
            torch.ones(1, 128, dtype=torch.bfloat16),
            qweight,
            scales,
            zeros,
            None,
            **arguments,
        )


@pytest.mark.parametrize("target", ["qweight", "scales", "zeros", "bias"])
def test_lpu_functional_rejects_inconsistent_tensor_shapes(target):
    qweight, scales, zeros = _canonical_tensors()
    tensors = {
        "qweight": qweight,
        "scales": scales,
        "zeros": zeros,
        "bias": torch.zeros(6, dtype=torch.bfloat16),
    }
    tensors[target] = tensors[target][:-1]

    with pytest.raises(ValueError, match=target):
        lpu_functional_linear(
            torch.ones(1, 128, dtype=torch.bfloat16),
            tensors["qweight"],
            tensors["scales"],
            tensors["zeros"],
            tensors["bias"],
            out_features=6,
            in_features=128,
            group_size=GROUP_SIZE,
            packing=PackingFormat.CANONICAL,
        )


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("qweight_dtype", TypeError, "uint8"),
        ("zeros_dtype", TypeError, "uint8"),
        ("scales_dtype", TypeError, "FP16"),
        ("nonfinite_scale", ValueError, "finite"),
        ("nonpositive_scale", ValueError, "positive"),
    ],
)
def test_lpu_functional_rejects_malformed_metadata(mutation, error, message):
    qweight, scales, zeros = _canonical_tensors()
    if mutation == "qweight_dtype":
        qweight = qweight.to(torch.int8)
    elif mutation == "zeros_dtype":
        zeros = zeros.to(torch.int8)
    elif mutation == "scales_dtype":
        scales = scales.to(torch.float32)
    elif mutation == "nonfinite_scale":
        scales[0, 0] = float("nan")
    elif mutation == "nonpositive_scale":
        scales[0, 0] = 0

    with pytest.raises(error, match=message):
        lpu_functional_linear(
            torch.ones(1, 128, dtype=torch.bfloat16),
            qweight,
            scales,
            zeros,
            None,
            out_features=6,
            in_features=128,
            group_size=GROUP_SIZE,
            packing=PackingFormat.CANONICAL,
        )
