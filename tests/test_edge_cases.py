"""Rarely used branches and error messages that no other test reaches."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

from shapiq_probly import (
    ConformalSetSizeGame,
    KNNDistanceScore,
    MarginalSampler,
    Measure,
    UncertaintyGame,
    entropy_decomposition,
    explain_ood,
    explain_uncertainty,
    from_members,
    from_sklearn,
    mc_dropout_sklearn,
    moebius_transform,
    plot_uncertainty_profile,
)
from shapiq_probly.dropout import _activate
from shapiq_probly.explanation import format_number
from shapiq_probly.measures import upper_entropy
from shapiq_probly.plots import _signed_short
from shapiq_probly.probly_bridge import from_probly

# --- adapters ------------------------------------------------------------


def test_member_that_never_saw_a_class_gets_probability_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 2))
    y = np.repeat([0, 1, 2], 30)
    full = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    partial = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X[:60], y[:60])  # classes 0 and 1 only
    probs = from_members([full, partial], classes=np.array([0, 1, 2]))(X[:8])
    assert probs.shape == (2, 8, 3)
    np.testing.assert_allclose(probs[1, :, 2], 0.0)
    np.testing.assert_allclose(probs[1, :, :2], partial.predict_proba(X[:8]))
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)


def test_regressors_with_and_without_their_own_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=60)
    bayes, plain = BayesianRidge().fit(X, y), LinearRegression().fit(X, y)
    out = from_members([bayes, plain])(X[:5])
    assert out.shape == (2, 5, 2)
    mean, std = bayes.predict(X[:5], return_std=True)
    np.testing.assert_allclose(out[0], np.column_stack([mean, std**2]))
    np.testing.assert_allclose(out[1, :, 1], 0.0)  # no variance reported: counts as 0


def test_adapter_error_messages():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    classifier = LogisticRegression().fit(X, (X[:, 0] > 0).astype(int))
    regressor = LinearRegression().fit(X, X[:, 0])
    with pytest.raises(TypeError, match="no 'estimators_'"):
        from_sklearn(classifier)
    with pytest.raises(TypeError, match="All ensemble members need 'predict_proba'"):
        from_members([classifier, regressor])
    with pytest.raises(TypeError, match="'predict_proba' \\(classification\\) or 'predict'"):
        from_members([object()])
    two_outputs = LinearRegression().fit(X, np.column_stack([X[:, 0], X[:, 1]]))
    with pytest.raises(ValueError, match="single-output"):
        from_members([two_outputs])(X[:3])


# --- games ---------------------------------------------------------------


def test_regression_members_as_single_column(toy_regressor, toy_background):
    x = toy_background[0]
    means_only = lambda X: toy_regressor(X)[..., 0]  # noqa: E731
    column = lambda X: toy_regressor(X)[..., :1]  # noqa: E731
    full = np.ones((1, 4), dtype=bool)
    kwargs = {"measure": "variance", "n_samples": 20}
    a = UncertaintyGame(means_only, x, toy_background, **kwargs).evaluate(full)
    b = UncertaintyGame(column, x, toy_background, **kwargs).evaluate(full)
    for notion in ("total", "aleatoric", "epistemic"):
        np.testing.assert_allclose(a[notion], b[notion])
    np.testing.assert_allclose(b["aleatoric"], 0.0, atol=1e-12)


def test_game_error_messages(toy_model, toy_background):
    x = toy_background[0]
    game = UncertaintyGame(toy_model, x, toy_background, n_samples=10)
    with pytest.raises(ValueError, match="Unknown notion"):
        game.value_function(np.ones((1, 4), dtype=bool), "certainty")
    with pytest.raises(ValueError, match="must return shape"):
        UncertaintyGame(lambda X: np.zeros((3, len(X))), x, toy_background, measure="entropy")
    with pytest.raises(ValueError, match="must return shape"):
        UncertaintyGame(lambda X: np.zeros((3, len(X), 4)), x, toy_background, measure="variance")
    with pytest.raises(ValueError, match="does not return class probabilities"):
        UncertaintyGame(lambda X: np.full((3, len(X), 2), 2.0), x, toy_background, measure="entropy")
    with pytest.raises(ValueError, match="2D array"):
        MarginalSampler(np.zeros(5))


def test_conformal_and_ood_error_messages(toy_background):
    y = (toy_background[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match=r"\(M, n, C\)"):
        ConformalSetSizeGame(lambda X: np.zeros((3, len(X))), toy_background[0], toy_background, (toy_background, y))
    with pytest.raises(ValueError, match="at least two reference points"):
        KNNDistanceScore(np.zeros((1, 4)))


# --- dropout, exact, measures --------------------------------------------


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_dropout_error_messages():
    with pytest.raises(ValueError, match="Unsupported activation"):
        _activate(np.zeros(3), "swish")
    X = np.random.default_rng(0).normal(size=(40, 2))
    two_outputs = MLPRegressor(hidden_layer_sizes=(4,), max_iter=20).fit(X, X)
    with pytest.raises(ValueError, match="multi-output"):
        mc_dropout_sklearn(two_outputs)


def test_moebius_needs_power_of_two():
    with pytest.raises(ValueError, match="2\\^n"):
        moebius_transform(np.zeros(3))


def test_upper_entropy_of_identical_members_is_their_entropy():
    p = np.tile(np.array([0.5, 0.3, 0.2]), (4, 6, 1))  # (M, n, C): the solver converges at once
    np.testing.assert_allclose(upper_entropy(p), entropy_decomposition(p)[0], atol=1e-9)


def test_custom_measure_as_function(toy_model, toy_background):
    def halved(probs):
        return tuple(0.5 * part for part in entropy_decomposition(probs))

    kwargs = {"n_samples": 20, "index": "SV"}
    custom = explain_uncertainty(toy_model, toy_background[0], toy_background, measure=halved, **kwargs)
    builtin = explain_uncertainty(toy_model, toy_background[0], toy_background, **kwargs)
    np.testing.assert_allclose(custom.shapley["total"], 0.5 * builtin.shapley["total"])
    as_object = Measure("halved", "classification", halved, "nats")
    wrapped = explain_uncertainty(toy_model, toy_background[0], toy_background, measure=as_object, **kwargs)
    np.testing.assert_allclose(wrapped.shapley["total"], custom.shapley["total"])


# --- explanation text and plots ------------------------------------------


def test_number_formatting():
    assert format_number(6660.4) == "6,660"
    assert format_number(0.01234) == "0.0123"
    assert _signed_short(1234.0) == "+1,234"
    assert _signed_short(-42.4) == "-42"


def test_summary_says_when_the_model_is_confident(toy_background):
    def sure(X):
        return np.tile(np.array([0.999, 0.001]), (3, len(X), 1))

    expl = explain_uncertainty(sure, toy_background[0], toy_background, n_samples=10)
    assert "the model is confident here" in expl.summary()


def test_summary_says_when_missing_knowledge_dominates(toy_background):
    def split(X):  # members contradict each other, each one being sure
        p = np.array([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]])
        return np.repeat(p[:, None, :], len(X), axis=1)

    expl = explain_uncertainty(split, toy_background[0], toy_background, n_samples=10)
    assert "mostly missing knowledge" in expl.summary()


@pytest.mark.filterwarnings("ignore:.*max_exact_features:UserWarning")
def test_summary_flags_approximated_values(toy_model, toy_background):
    x = toy_background[0]
    expl = explain_uncertainty(toy_model, x, toy_background, n_samples=10, max_exact_features=2, budget=30)
    assert "values are approximated" in expl.summary()
    scalar = explain_ood(x, toy_background, n_samples=10, max_exact_features=2, budget=30)
    assert "values are approximated" in scalar.summary()


def test_profile_with_many_features_names_the_unlabelled_ones():
    rng = np.random.default_rng(0)
    background = rng.normal(size=(100, 8))

    def members(X):
        logits = np.stack([3 * X[:, 0], 3 * X[:, 0] + 2 * X[:, 1], -3 * X[:, 0]])
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1 - p, p], axis=-1)

    expl = explain_uncertainty(members, background[0], background, n_samples=10, index="SV")
    fig = plot_uncertainty_profile(expl)
    notes = [t.get_text() for t in fig.axes[0].texts if t.get_text().startswith("near zero")]
    assert len(notes) == 1
    assert "x7" in notes[0]
    assert "x0" not in notes[0]
    plt.close(fig)


# --- probly bridge -------------------------------------------------------


def test_probly_bridge_accepts_tensor_like_and_rejects_text():
    class TensorLike:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Rep:
        def __init__(self, out):
            self.out = out

        def represent(self, X):
            return self.out

    probs = np.full((4, 3, 2), 0.5)  # (n, M, C)
    assert from_probly(Rep(TensorLike(probs)))(np.zeros((4, 2))).shape == (3, 4, 2)
    with pytest.raises(TypeError, match="Cannot convert"):
        from_probly(Rep("not numbers"))(np.zeros((4, 2)))
