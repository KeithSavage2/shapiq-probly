"""Bridge to probly. The first tests use a stand-in; the rest need probly installed."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from shapiq_probly import UncertaintyGame, explain_uncertainty, make_uncertainty_benchmark
from shapiq_probly.measures import entropy_decomposition
from shapiq_probly.probly_bridge import from_probly


class _FakeSample:
    """Mimics a probly sample: batch first, members on ``sample_axis``."""

    def __init__(self, probs_nmc: np.ndarray) -> None:
        self.sample_axis = 1
        self.array = type("Dist", (), {"probabilities": probs_nmc})()


class _FakeRepresenter:
    def __init__(self, wrap: bool) -> None:
        self.wrap = wrap

    def represent(self, X: np.ndarray) -> object:
        p = 1.0 / (1.0 + np.exp(-np.stack([X[:, 0], 2 * X[:, 1], X[:, 0] - X[:, 1]], axis=1)))  # (n, M)
        probs = np.stack([1 - p, p], axis=-1)  # (n, M, C)
        return _FakeSample(probs) if self.wrap else probs


@pytest.mark.parametrize("wrap", [True, False])
def test_members_come_first(wrap):
    X = np.random.default_rng(0).normal(size=(5, 2))
    probs = from_probly(_FakeRepresenter(wrap))(X)
    assert probs.shape == (3, 5, 2)
    np.testing.assert_allclose(probs[1, :, 1], 1.0 / (1.0 + np.exp(-2 * X[:, 1])))


def test_model_without_members_is_rejected():
    class NoMembers:
        def represent(self, X):
            return np.full((len(X), 2), 0.5)

    with pytest.raises(ValueError, match="no members"):
        from_probly(NoMembers())(np.zeros((4, 2)))


def test_to_input_and_to_probs_are_used():
    seen = {}

    class Rep:
        def represent(self, X):
            seen["type"] = type(X)
            return "raw"

    predict = from_probly(Rep(), to_probs=lambda r: np.full((2, 3, 2), 0.5), to_input=lambda X: X.tolist())
    assert predict(np.zeros((3, 2))).shape == (2, 3, 2)
    assert seen["type"] is list


# --- real probly ---------------------------------------------------------


@pytest.fixture(scope="module")
def probly_setup():
    pytest.importorskip("probly")
    from probly.method.ensemble import ensemble  # noqa: PLC0415
    from probly.representer import representer  # noqa: PLC0415

    data = make_uncertainty_benchmark(n_samples=600, random_state=0)
    members = ensemble(LogisticRegression(max_iter=500), num_members=5)
    rng = np.random.default_rng(0)
    for member in members:
        rows = rng.integers(0, len(data.X), len(data.X))
        member.fit(data.X[rows], data.y[rows])
    return data, members, representer(members)


def test_matches_member_predictions(probly_setup):
    data, members, rep = probly_setup
    X = data.X[:7]
    expected = np.stack([m.predict_proba(X) for m in members])
    np.testing.assert_allclose(from_probly(rep)(X), expected, atol=1e-12)


def test_full_coalition_equals_probly_quantify(probly_setup):
    from probly.quantification import quantify  # noqa: PLC0415

    data, _, rep = probly_setup
    x = data.X[3]
    game = UncertaintyGame(from_probly(rep), x, data.X[:50], n_samples=10, random_state=0)
    full = np.ones((1, data.X.shape[1]), dtype=bool)
    reference = quantify(rep.represent(x[None, :]))
    for kind in ("total", "aleatoric", "epistemic"):
        ours = game.value_function(full, kind)[0]
        np.testing.assert_allclose(ours, np.asarray(getattr(reference, kind))[0], atol=1e-9)


def test_entropy_measure_equals_probly(probly_setup):
    from probly.quantification import quantify  # noqa: PLC0415

    data, _, rep = probly_setup
    out = rep.represent(data.X[:20])
    reference = quantify(out)
    ours = entropy_decomposition(from_probly(rep)(data.X[:20]))
    for values, kind in zip(ours, ("total", "aleatoric", "epistemic"), strict=True):
        np.testing.assert_allclose(values, np.asarray(getattr(reference, kind)), atol=1e-9)


def test_explanation_is_additive(probly_setup):
    data, _, rep = probly_setup
    expl = explain_uncertainty(from_probly(rep), data.X[3], data.X[:50], n_samples=10, random_state=0)
    assert expl.check_additivity()


def test_plain_forest_gives_helpful_error(probly_setup):
    from probly.representer import representer  # noqa: PLC0415

    data, _, _ = probly_setup
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(data.X, data.y)
    with pytest.raises(ValueError, match="no members"):
        from_probly(representer(forest))(data.X[:4])
