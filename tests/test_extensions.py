"""Credal sets, conformal set size, OOD score and custom scalar games."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from shapiq_probly import (
    ConformalSetSizeGame,
    KNNDistanceScore,
    OODGame,
    ScalarExplanation,
    ScalarGame,
    UncertaintyGame,
    all_coalitions,
    explain_game,
    explain_ood,
    explain_set_size,
    explain_uncertainty,
    make_uncertainty_benchmark,
    plot_interaction_network,
    plot_scalar_explanation,
)
from shapiq_probly.conformal import conformal_quantile
from shapiq_probly.synthetic import FEATURE_NAMES

NOISE, PAIR_A, PAIR_B, DUMMY = range(4)
X0 = np.array([0.8, 0.8, 1.2, 0.5])

# ---------------------------------------------------------------- credal sets


@pytest.mark.parametrize("aggregation", ["member_mean", "pointwise", "pooled"])
def test_credal_game_is_additive(toy_model, toy_background, aggregation):
    game = UncertaintyGame(toy_model, X0, toy_background, measure="credal", aggregation=aggregation, n_samples=30)
    ev = game.evaluate(all_coalitions(4))
    np.testing.assert_allclose(ev["total"], ev["aleatoric"] + ev["epistemic"], atol=1e-12)
    assert (ev["epistemic"] >= -1e-12).all()


def test_credal_explanation(toy_model, toy_background, benchmark, forest):
    expl = explain_uncertainty(toy_model, X0, toy_background, measure="credal", n_samples=30)
    assert expl.config["measure"] == "credal"
    assert expl.check_additivity()
    entropy_based = explain_uncertainty(toy_model, X0, toy_background, measure="entropy", n_samples=30)
    assert expl.uncertainty["epistemic"] >= entropy_based.uncertainty["epistemic"] - 1e-9
    # On the benchmark the credal set is wider in the gap than where the model is certain.
    width = {
        name: explain_uncertainty(forest, benchmark.queries[name], benchmark.X, measure="credal", index="SV")
        for name in ("gap", "certain")
    }
    assert width["gap"].uncertainty["epistemic"] > width["certain"].uncertainty["epistemic"]


def test_credal_is_classification_only(reg_forest, reg_benchmark):
    with pytest.raises(ValueError, match="regression model"):
        explain_uncertainty(reg_forest, reg_benchmark.X[0], reg_benchmark.X, measure="credal")


# ------------------------------------------------------- conformal prediction


def test_conformal_quantile():
    scores = np.arange(1, 11) / 10
    assert conformal_quantile(scores, alpha=0.2) == pytest.approx(0.9)  # ceil(11 * 0.8) = 9th smallest
    assert conformal_quantile(scores, alpha=0.5) == pytest.approx(0.6)
    assert conformal_quantile(scores, alpha=0.01) == float("inf")  # too few calibration points


def test_conformal_coverage_and_set_at_x(benchmark, forest, calibration):
    test = make_uncertainty_benchmark(n_samples=3000, random_state=7)
    game = ConformalSetSizeGame(forest, benchmark.queries["gap"], benchmark.X, calibration, alpha=0.1)
    proba = forest.predict_proba(test.X)
    covered = proba[np.arange(len(test.y)), test.y] >= game.threshold
    assert covered.mean() > 0.87
    assert game.value_function(np.ones((1, 4), dtype=bool))[0] == len(game.prediction_set)
    assert game.set_sizes(benchmark.queries["gap"])[0] == len(game.prediction_set)


def test_conformal_aggregations(benchmark, forest, calibration):
    x = benchmark.queries["noisy"]
    pointwise = ConformalSetSizeGame(forest, x, benchmark.X, calibration, n_samples=50)
    empty = np.zeros((1, 4), dtype=bool)
    draws = pointwise.sampler.sample(x, empty[0])
    assert pointwise.value_function(empty)[0] == pytest.approx(pointwise.set_sizes(draws).mean())
    pooled = ConformalSetSizeGame(forest, x, benchmark.X, calibration, n_samples=50, aggregation="pooled")
    values = pooled.value_function(all_coalitions(4))
    np.testing.assert_array_equal(values, np.round(values))
    with pytest.raises(ValueError, match="Unknown aggregation"):
        ConformalSetSizeGame(forest, x, benchmark.X, calibration, aggregation="member_mean")
    with pytest.raises(ValueError, match="alpha"):
        ConformalSetSizeGame(forest, x, benchmark.X, calibration, alpha=1.5)


def test_conformal_handles_arbitrary_class_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = np.where(X[:, 0] + 0.5 * rng.normal(size=600) > 0, "yes", "no")
    model = RandomForestClassifier(n_estimators=20, min_samples_leaf=5, random_state=0).fit(X[:300], y[:300])
    game = ConformalSetSizeGame(model, X[0], X[:300], (X[300:], y[300:]), alpha=0.1, n_samples=20)
    assert set(game.prediction_set) <= {"yes", "no"}
    with pytest.raises(ValueError, match="not in 'classes'"):
        ConformalSetSizeGame(model, X[0], X[:300], (X[300:], np.full(300, "maybe")), n_samples=20)


def test_set_size_explanation_is_efficient(benchmark, forest, calibration):
    expl = explain_set_size(
        forest, benchmark.queries["gap"], benchmark.X, calibration, imputer="knn", feature_names=list(FEATURE_NAMES)
    )
    assert isinstance(expl, ScalarExplanation)
    assert expl.value == len(expl.info["prediction_set"])
    assert sum(expl.attribution.values.values()) == pytest.approx(expl.value - expl.baseline)
    assert expl.shapley.sum() == pytest.approx(expl.value - expl.baseline)
    assert f"prediction_set: {expl.info['prediction_set']}" in expl.summary()


def test_noise_feature_is_what_puts_both_classes_into_the_set(benchmark, forest, calibration):
    """Ground truth on average: two-class sets are mostly caused by label noise.

    A single point is not a stable test here: with two classes the set size is
    1 or 2 and flips with the seed for borderline points such as the gap query.
    """
    test = make_uncertainty_benchmark(n_samples=1500, random_state=7)
    game = ConformalSetSizeGame(forest, test.X[0], benchmark.X, calibration)
    two_classes = np.flatnonzero(game.set_sizes(test.X) == 2)[:25]
    assert len(two_classes) == 25
    shapley = np.array(
        [
            explain_set_size(forest, test.X[i], benchmark.X, calibration, index="SV", n_samples=40).shapley
            for i in two_classes
        ]
    )
    mean = shapley.mean(axis=0)
    assert mean.argmax() == NOISE
    assert mean[NOISE] > 1.5 * np.delete(mean, NOISE).max()
    assert abs(mean[DUMMY]) < 0.15


# ------------------------------------------------------------------ OOD score


def test_knn_score_does_not_count_a_point_as_its_own_neighbour():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 3))
    score = KNNDistanceScore(reference, k=5)
    standardised = (reference - reference.mean(axis=0)) / reference.std(axis=0)
    dist = np.linalg.norm(standardised[:10, None, :] - standardised[None, :, :], axis=-1)
    expected = np.sort(dist, axis=1)[:, 1:6].mean(axis=1)  # skip the zero distance to itself
    np.testing.assert_allclose(score(reference[:10]), expected)
    assert score(np.array([[8.0, 8.0, 8.0]]))[0] > 5 * score(reference).mean()


def test_ood_game_values(benchmark):
    x = np.array([0.3, 0.2, 0.3, 3.0])
    game = OODGame(x, benchmark.X, n_samples=50)
    assert game.value_function(np.ones((1, 4), dtype=bool))[0] == pytest.approx(game.score(x[None, :])[0])
    draws = game.sampler.sample(x, np.zeros(4, dtype=bool))
    assert game.value_function(np.zeros((1, 4), dtype=bool))[0] == pytest.approx(game.score(draws).mean())
    with pytest.raises(ValueError, match="Unknown score"):
        OODGame(x, benchmark.X, score="nope")


@pytest.mark.parametrize("imputer", ["marginal", "knn"])
def test_shifted_feature_and_unseen_combination_are_found(benchmark, imputer):
    names = list(FEATURE_NAMES)
    shifted = explain_ood(np.array([0.3, 0.2, 0.3, 3.0]), benchmark.X, imputer=imputer, feature_names=names)
    assert shifted.shapley.argmax() == DUMMY
    assert shifted.shapley[DUMMY] > 10 * np.abs(np.delete(shifted.shapley, DUMMY)).max()

    gap = explain_ood(benchmark.queries["gap"], benchmark.X, imputer=imputer, feature_names=names)
    assert gap.value > 1.5 * gap.baseline
    assert gap.top_interactions(1)[0][0] == "pair_a x pair_b"
    assert gap.top_interactions(1)[0][1] > 0.2

    typical = explain_ood(benchmark.queries["certain"], benchmark.X, imputer=imputer, feature_names=names)
    assert np.abs(typical.shapley).max() < 0.15


def test_custom_ood_score(benchmark):
    forest = IsolationForest(n_estimators=50, random_state=0).fit(benchmark.X)

    def isolation_score(X):
        return -forest.score_samples(X)

    expl = explain_ood(np.array([0.3, 0.2, 0.3, 3.0]), benchmark.X, score=isolation_score, n_samples=50, index="SV")
    assert expl.info["score"] == "isolation_score"
    assert expl.shapley.argmax() == DUMMY


# -------------------------------------------------------------- scalar games


def test_linear_scalar_game_has_closed_form_shapley_values(toy_background):
    weights = np.array([2.0, -1.0, 0.5, 0.0])
    game = ScalarGame(X0, toy_background, lambda points: float((points @ weights).mean()), name="linear", n_samples=40)
    expl = explain_game(game)
    draws = game.sampler.sample(X0, np.zeros(4, dtype=bool))
    np.testing.assert_allclose(expl.shapley, weights * (X0 - draws.mean(axis=0)), atol=1e-10)
    assert max(abs(v) for v in expl.attribution.order(2).values()) < 1e-10
    assert expl.name == "linear"
    with pytest.raises(NotImplementedError):
        ScalarGame(X0, toy_background).value_function(np.ones((1, 4), dtype=bool))


def test_scalar_explanation_roundtrip_approximation_and_plots(toy_background, tmp_path):
    def value(points):
        return float(np.tanh(points[:, 0] * points[:, 1] + points[:, 2]).mean())

    exact = explain_game(ScalarGame(X0, toy_background, value, n_samples=30), feature_names=list("abcd"))
    exact.save(tmp_path / "s.json")
    loaded = ScalarExplanation.load(tmp_path / "s.json")
    np.testing.assert_allclose(loaded.shapley, exact.shapley)
    assert loaded.summary() == exact.summary()
    with pytest.warns(UserWarning, match="approximating"):
        approx = explain_game(ScalarGame(X0, toy_background, value, n_samples=30), max_exact_features=3, budget=800)
    assert approx.attribution.estimated
    np.testing.assert_allclose(approx.shapley, exact.shapley, atol=0.03)
    for plot in (plot_scalar_explanation, plot_interaction_network):
        fig = plot(exact)
        fig.savefig(tmp_path / "p.png")
        plt.close(fig)
