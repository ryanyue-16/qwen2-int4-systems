import pytest
import torch

from qwen_int4.validation import ActivationRecorder, tensor_metrics


def test_tensor_metrics_for_identical_values():
    value = torch.tensor([[1.0, -2.0, 3.0]])
    metrics = tensor_metrics(value, value.clone())
    assert metrics.max_abs == 0
    assert metrics.mean_abs == 0
    assert metrics.cosine == pytest.approx(1.0)


def test_tensor_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        tensor_metrics(torch.zeros(2), torch.zeros(3))


def test_activation_recorder_captures_named_module():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())
    with ActivationRecorder(model, {"0"}) as recorder:
        model(torch.ones(1, 3))
    assert recorder.activations["0"].shape == (1, 2)
