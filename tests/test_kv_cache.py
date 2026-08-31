"""Correctness tests for Phase F prefill/decode KV cache behavior."""

from types import SimpleNamespace

import torch
import pytest

from qwen_int4.linear import QuantLinear
from qwen_int4.generation import CachedGenerationState, greedy_generate_cached
from qwen_int4.model import QwenInt4ForCausalLM
from qwen_int4.quantization import PackingFormat, quantize_asymmetric_rtn


def _model() -> QwenInt4ForCausalLM:
    config = SimpleNamespace(
        hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        intermediate_size=16, num_hidden_layers=2, vocab_size=32, rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    torch.manual_seed(123)
    model = QwenInt4ForCausalLM(config, packing=PackingFormat.CANONICAL, group_size=4)
    for module in model.modules():
        if isinstance(module, QuantLinear):
            qweight, scales, zeros = quantize_asymmetric_rtn(
                torch.randn(module.out_features, module.in_features) * 0.1,
                group_size=4, packing=PackingFormat.CANONICAL,
            )
            module.qweight.copy_(qweight)
            module.scales.copy_(scales)
            module.zeros.copy_(zeros)
    return model.eval()


def test_prefill_then_token_decode_matches_full_forward():
    model = _model()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    full = model(tokens)
    prefill, cache = model(tokens[:, :3], use_cache=True)
    decode_1, cache = model(tokens[:, 3:4], past_key_values=cache, use_cache=True)
    decode_2, cache = model(tokens[:, 4:5], past_key_values=cache, use_cache=True)
    actual = torch.cat((prefill, decode_1, decode_2), dim=1)
    torch.testing.assert_close(actual, full, rtol=1e-3, atol=1e-3)
    assert len(cache) == 2
    assert cache[0][0].shape == (1, 1, 5, 4)


def test_batched_prefill_then_multi_token_decode_matches_full_forward():
    model = _model()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    full = model(tokens)
    prefill, cache = model(tokens[:, :2], use_cache=True)
    decoded, cache = model(tokens[:, 2:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(torch.cat((prefill, decoded), dim=1), full, rtol=1e-3, atol=1e-3)
    assert cache[1][1].shape == (2, 1, 4, 4)


def test_cached_greedy_generation_matches_prefix_recomputation():
    model = _model()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected = prompt.clone()
    for _ in range(4):
        expected = torch.cat((expected, model(expected)[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
    actual = greedy_generate_cached(model, prompt, max_new_tokens=4)
    torch.testing.assert_close(actual, expected)


def test_left_padded_batch_generation_matches_each_unpadded_request():
    model = _model()
    padded = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.bool)
    batched = greedy_generate_cached(model, padded, attention_mask=mask, max_new_tokens=2)
    first = greedy_generate_cached(model, torch.tensor([[1, 2, 3]]), max_new_tokens=2)
    second = greedy_generate_cached(model, torch.tensor([[4, 5, 6, 7]]), max_new_tokens=2)
    torch.testing.assert_close(batched[0, -5:], first[0])
    torch.testing.assert_close(batched[1, -6:], second[0])


def test_generation_state_rejects_right_padding_and_all_padding_rows():
    model = _model()
    with pytest.raises(ValueError, match="left-padded"):
        CachedGenerationState.prefill(
            model, torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]], dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="non-empty"):
        CachedGenerationState.prefill(
            model, torch.tensor([[0, 0, 0]]), torch.tensor([[0, 0, 0]], dtype=torch.bool)
        )


def test_generation_state_reorders_removes_and_closes_owned_cache():
    model = _model()
    state = CachedGenerationState.prefill(
        model, torch.tensor([[1, 2], [3, 4], [5, 6]])
    )
    state.select_requests([2, 0])
    assert state.generated_ids.tolist() == [[5, 6], [1, 2]]
    assert state.past_key_values[0][0].shape[0] == 2
    state.decode(state.next_greedy_tokens())
    state.close()
    assert state.past_key_values == ()
    with pytest.raises(RuntimeError, match="closed"):
        state.next_greedy_tokens()


def test_generation_state_rejects_duplicate_request_indices():
    state = CachedGenerationState.prefill(_model(), torch.tensor([[1], [2]]))
    with pytest.raises(ValueError, match="beam-cache"):
        state.select_requests([0, 0])
