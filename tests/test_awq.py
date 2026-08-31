import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qwen_int4.awq import (
    fake_quantize_asymmetric,
    search_awq_scales_by_module_output,
    search_gqa_v_to_o_scales,
    search_awq_scales,
    search_groupwise_clipping,
)
from qwen_int4.calibration import ActivationCollector, sequential_block_calibration
from qwen_int4.transforms import (
    ModuleStateSnapshot,
    expand_qwen2_gqa_scales,
    qwen2_awq_mappings,
    scale_linear_to_linear,
    scale_norm_to_linears,
)


class ChannelNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(channels) + 0.5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.weight


def test_activation_collector_is_channelwise_and_memory_bounded():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    values = torch.randn(2, 5, 4)
    collector = ActivationCollector(max_cached_tokens=3)
    collector.attach(model)
    model(values)
    collector.detach()

    statistics = collector.statistics()
    samples = collector.samples()
    assert statistics["0"].count == 10
    torch.testing.assert_close(statistics["0"].mean_abs, values.abs().mean(dim=(0, 1)))
    assert samples["0"].shape == (3, 4)
    assert samples["2"].shape == (3, 3)


def test_activation_collector_reservoir_is_deterministic_and_covers_late_tokens():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    first = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    second = torch.arange(5, 50, dtype=torch.float32).view(1, 45, 1)

    collected = []
    for _ in range(2):
        collector = ActivationCollector(
            max_cached_tokens=4, sampling_strategy="reservoir", sample_seed=42
        )
        collector.attach(model)
        model(first)
        model(second)
        collector.detach()
        collected.append(collector.samples()[""])

    torch.testing.assert_close(collected[0], collected[1])
    assert collected[0].shape == (4, 1)
    assert collected[0].max().item() > first.max().item()


def test_sequential_calibration_releases_each_block_hook():
    blocks = [nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(2)]
    values = torch.randn(1, 3, 4)
    statistics, output = sequential_block_calibration(
        blocks,
        values,
        forward_block=lambda block, hidden: block(hidden),
        max_cached_tokens=2,
    )

    assert len(statistics) == 2
    assert statistics[0]["0"].count == 3
    assert output.shape == values.shape
    assert all(not child._forward_hooks for block in blocks for child in block.modules())


def test_norm_to_multiple_linears_is_function_equivalent():
    torch.manual_seed(1)
    norm = ChannelNorm(6)
    linears = [nn.Linear(6, 4), nn.Linear(6, 3)]
    values = torch.randn(2, 5, 6)
    expected = [linear(norm(values)) for linear in linears]

    scale_norm_to_linears(norm, linears, torch.rand(6) + 0.25)
    actual = [linear(norm(values)) for linear in linears]

    for reference, candidate in zip(expected, actual):
        torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)


def test_linear_to_linear_is_function_equivalent_with_bias():
    torch.manual_seed(2)
    source = nn.Linear(5, 7, bias=True)
    target = nn.Linear(7, 3, bias=False)
    values = torch.randn(4, 5)
    expected = target(source(values))

    scale_linear_to_linear(source, target, torch.rand(7) + 0.25)

    torch.testing.assert_close(target(source(values)), expected, rtol=1e-5, atol=1e-6)


def test_swiglu_up_to_down_scale_is_function_equivalent():
    torch.manual_seed(3)
    gate = nn.Linear(5, 8, bias=False)
    up = nn.Linear(5, 8, bias=False)
    down = nn.Linear(8, 5, bias=False)
    values = torch.randn(2, 4, 5)
    expected = down(F.silu(gate(values)) * up(values))

    scale_linear_to_linear(up, down, torch.rand(8) + 0.25)
    actual = down(F.silu(gate(values)) * up(values))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gqa_v_to_o_scale_expansion_is_function_equivalent():
    torch.manual_seed(4)
    num_heads, num_kv_heads, head_dim = 4, 2, 3
    v_proj = nn.Linear(6, num_kv_heads * head_dim, bias=True)
    o_proj = nn.Linear(num_heads * head_dim, 6, bias=False)
    values = torch.randn(2, 5, 6)

    def attention_value_path() -> torch.Tensor:
        projected = v_proj(values).view(2, 5, num_kv_heads, head_dim)
        repeated = projected.repeat_interleave(num_heads // num_kv_heads, dim=2)
        return o_proj(repeated.reshape(2, 5, -1))

    expected = attention_value_path()
    scales = torch.rand(num_kv_heads * head_dim) + 0.25
    expanded = expand_qwen2_gqa_scales(
        scales,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )
    scale_linear_to_linear(
        v_proj, o_proj, scales, expanded_target_scales=expanded
    )

    torch.testing.assert_close(attention_value_path(), expected, rtol=1e-5, atol=1e-6)


def test_gqa_v_to_o_search_returns_base_kv_channels_and_preserves_modules():
    torch.manual_seed(40)
    projection = nn.Linear(8, 5, bias=False)
    original = projection.weight.detach().clone()
    inputs = torch.randn(2, 3, 8)

    result = search_gqa_v_to_o_scales(
        inputs,
        projection,
        num_attention_heads=4,
        num_key_value_heads=2,
        group_size=4,
        grid_points=6,
    )

    assert result.scales.shape == (4,)
    assert torch.all(result.scales > 0)
    assert result.loss <= result.identity_loss
    torch.testing.assert_close(projection.weight, original)


def test_module_snapshot_restores_state_after_failure():
    module = nn.Linear(4, 3)
    original = copy.deepcopy(module.state_dict())

    with pytest.raises(RuntimeError), ModuleStateSnapshot([module]):
        module.weight.data.zero_()
        raise RuntimeError("candidate failed")

    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_awq_search_does_not_mutate_weights_and_never_loses_to_identity():
    torch.manual_seed(5)
    linears = [nn.Linear(8, 6), nn.Linear(8, 4)]
    original = [linear.weight.detach().clone() for linear in linears]
    inputs = torch.randn(3, 7, 8)

    result = search_awq_scales(
        inputs, linears, group_size=4, grid_points=8, duo_scaling=True
    )

    assert result.scales.shape == (8,)
    assert torch.all(result.scales > 0)
    assert result.loss <= result.identity_loss
    for linear, weight in zip(linears, original):
        torch.testing.assert_close(linear.weight, weight)


def test_module_output_awq_search_restores_weights_and_never_loses_to_identity():
    torch.manual_seed(6)
    first = nn.Linear(8, 6, bias=False)
    second = nn.Linear(8, 4, bias=False)
    original = [first.weight.detach().clone(), second.weight.detach().clone()]
    batches = (torch.randn(1, 5, 8), torch.randn(1, 3, 8))

    def forward_batch(_index: int, values: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.tanh(first(values)), second(values)), dim=-1)

    result = search_awq_scales_by_module_output(
        batches,
        (first, second),
        forward_batch,
        group_size=4,
        grid_points=6,
    )

    assert result.scales.shape == (8,)
    assert result.loss <= result.identity_loss
    torch.testing.assert_close(first.weight, original[0])
    torch.testing.assert_close(second.weight, original[1])


def test_asymmetric_fake_quantization_and_qwen2_mapping_inventory():
    weight = torch.tensor([[-3.0, -1.0, 0.5, 4.0]])
    restored = fake_quantize_asymmetric(weight, group_size=4)

    assert restored.shape == weight.shape
    assert torch.isfinite(restored).all()
    mappings = qwen2_awq_mappings()
    assert len(mappings) == 4
    assert mappings[0].targets == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    )


def test_groupwise_clipping_returns_independent_choices_without_mutation():
    torch.manual_seed(8)
    weight = torch.randn(3, 8)
    weight[:, 0] *= 20
    original = weight.clone()
    inputs = torch.randn(2, 6, 8)

    result = search_groupwise_clipping(
        weight, inputs, group_size=4, ratios=(1.0, 0.9, 0.8)
    )

    assert result.clip_max.shape == (3, 2)
    assert result.ratios.shape == (3, 2)
    assert torch.all((result.ratios >= 0.8) & (result.ratios <= 1.0))
    assert torch.isfinite(result.loss).all()
    torch.testing.assert_close(weight, original)
