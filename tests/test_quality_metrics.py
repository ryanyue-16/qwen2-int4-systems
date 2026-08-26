import pytest
import torch
from torch import nn

from qwen_int4.validation import (
    language_model_metrics,
    sliding_window_language_model_metrics,
)


def test_language_model_metrics_prefers_correct_next_token():
    input_ids = torch.tensor([[0, 1, 2]])
    good = torch.zeros(1, 3, 4)
    bad = torch.zeros(1, 3, 4)
    good[0, 0, 1] = 5
    good[0, 1, 2] = 5
    bad[0, 0, 3] = 5
    bad[0, 1, 3] = 5
    assert language_model_metrics(good, input_ids)["nll"] < language_model_metrics(bad, input_ids)["nll"]


class _PerfectNextTokenModel(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        vocabulary_size = 8
        logits = torch.full((*input_ids.shape, vocabulary_size), -20.0)
        next_ids = (input_ids + 1) % vocabulary_size
        return logits.scatter(-1, next_ids.unsqueeze(-1), 20.0)


def test_sliding_window_scores_each_target_once():
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    metrics, predictions, target_log_probs = sliding_window_language_model_metrics(
        _PerfectNextTokenModel(), input_ids, window_size=4, stride=2
    )

    assert metrics["predictions"] == input_ids.shape[1] - 1
    assert metrics["windows"] == 3
    assert metrics["perplexity"] < 1.01
    assert torch.equal(predictions, input_ids[0, 1:])
    assert target_log_probs.shape == predictions.shape
