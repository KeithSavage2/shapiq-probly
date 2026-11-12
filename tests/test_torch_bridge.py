"""Integration with torch. Skipped unless torch is installed (never run in development)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from shapiq_probly import explain_uncertainty  # noqa: E402
from shapiq_probly.torch_bridge import from_torch_dropout, from_torch_ensemble  # noqa: E402


def _net(p: float = 0.5, n_out: int = 3) -> "torch.nn.Module":
    return torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Dropout(p), torch.nn.Linear(16, n_out))


@pytest.fixture
def points():
    return np.random.default_rng(0).normal(size=(30, 4))


def test_ensemble(points):
    torch.manual_seed(0)
    predict = from_torch_ensemble([_net() for _ in range(4)])
    probs = predict(points)
    assert predict.task == "classification"
    assert probs.shape == (4, 30, 3)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    assert np.abs(probs - probs[0]).max() > 1e-3


def test_dropout_masks_are_fixed_members(points):
    torch.manual_seed(0)
    model = _net(p=0.5).train()
    predict = from_torch_dropout(model, n_samples=6)
    full = predict(points)
    assert full.shape == (6, 30, 3)
    np.testing.assert_allclose(full, predict(points), atol=1e-6)  # same masks on every call
    np.testing.assert_allclose(predict(points[:5]), full[:, :5], atol=1e-6)  # and for every row
    assert np.abs(full - full[0]).max() > 1e-3  # members differ
    assert model.training  # mode restored
    assert all(len(layer._forward_hooks) == 0 for layer in model.modules())  # hooks removed


def test_dropout_with_p_zero_is_the_network(points):
    torch.manual_seed(0)
    model = _net(p=0.0).eval()
    with torch.no_grad():
        expected = torch.softmax(model(torch.as_tensor(points, dtype=torch.float32)), dim=-1).numpy()
    np.testing.assert_allclose(from_torch_dropout(model, n_samples=2)(points)[1], expected, atol=1e-6)


def test_regression_output_and_explanation(points):
    torch.manual_seed(0)
    predict = from_torch_dropout(_net(p=0.3, n_out=1), n_samples=5, output="regression")
    assert predict.task == "regression"
    assert predict(points).shape == (5, 30)
    expl = explain_uncertainty(predict, points[0], points, n_samples=10)
    assert expl.task == "regression"
    assert expl.check_additivity(1e-6)


def test_model_without_dropout_is_rejected():
    with pytest.raises(ValueError, match=r"no torch\.nn\.Dropout"):
        from_torch_dropout(torch.nn.Linear(4, 2))
