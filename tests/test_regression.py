import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.tree import DecisionTreeRegressor

from shapiq_probly import (
    UncertaintyExplanation,
    UncertaintyGame,
    all_coalitions,
    as_member_predictor,
    explain_uncertainty,
    make_uncertainty_benchmark,
    plot_interaction_network,
    plot_uncertainty_explanation,
    plot_uncertainty_profile,
)
from shapiq_probly.games import AGGREGATIONS
from shapiq_probly.synthetic import FEATURE_NAMES, true_mean, true_noise_std

NOISE, PAIR_A, PAIR_B, DUMMY = range(4)
X0 = np.array([0.8, 0.8, 1.2, 0.5])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = X[:, 0] + rng.normal(scale=0.1 + np.abs(X[:, 1]), size=400)
    return X, y


def test_tree_members_report_leaf_mean_and_variance(data):
    X, y = data
    tree = DecisionTreeRegressor(min_samples_leaf=20, random_state=0).fit(X, y)
    out = as_member_predictor([tree])(X)
    assert out.shape == (1, 400, 2)
    leaves = tree.apply(X)
    for leaf in np.unique(leaves)[:5]:
        rows = leaves == leaf
        assert out[0, rows, 0] == pytest.approx(y[rows].mean())
        assert out[0, rows, 1] == pytest.approx(y[rows].var())


def test_forest_adapter(data):
    X, y = data
    model = RandomForestRegressor(n_estimators=12, min_samples_leaf=10, random_state=0).fit(X, y)
    predict = as_member_predictor(model)
    out = predict(X[:30])
    assert predict.task == "regression"
    assert out.shape == (12, 30, 2)
    np.testing.assert_allclose(out[..., 0].mean(axis=0), model.predict(X[:30]))
    assert (out[..., 1] > 0).all()


def test_bagging_feature_subsets_and_other_regressors(data):
    X, y = data
    bag = BaggingRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=8, max_features=2, random_state=0)
    out = as_member_predictor(bag.fit(X, y))(X[:10])
    np.testing.assert_allclose(out[..., 0].mean(axis=0), bag.predict(X[:10]))

    members = [LinearRegression().fit(X, y), BayesianRidge().fit(X, y)]
    out = as_member_predictor(members)(X[:10])
    assert out.shape == (2, 10, 2)
    assert (out[0, :, 1] == 0).all()  # no predictive variance available
    assert (out[1, :, 1] > 0).all()  # predict(..., return_std=True)

    with pytest.raises(TypeError, match="boosting"):
        as_member_predictor(GradientBoostingRegressor(n_estimators=3).fit(X, y))


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
@pytest.mark.parametrize("imputer", ["marginal", "gaussian", "knn"])
def test_additivity_for_every_coalition(toy_regressor, toy_background, aggregation, imputer):
    game = UncertaintyGame(toy_regressor, X0, toy_background, aggregation=aggregation, imputer=imputer, n_samples=40)
    assert game.task == "regression"
    assert game.measure.name == "variance"
    ev = game.evaluate(all_coalitions(4))
    np.testing.assert_allclose(ev["total"], ev["aleatoric"] + ev["epistemic"], atol=1e-10)
    assert (ev["aleatoric"] >= -1e-10).all()
    assert (ev["epistemic"] >= -1e-10).all()


def test_full_coalition_is_the_models_own_variance(toy_regressor, toy_background):
    game = UncertaintyGame(toy_regressor, X0, toy_background, n_samples=40)
    ev = game.evaluate(np.ones((1, 4), dtype=bool))
    own = toy_regressor(X0[None, :])[:, 0, :]
    assert ev["aleatoric"][0] == pytest.approx(own[:, 1].mean())
    assert ev["epistemic"][0] == pytest.approx(own[:, 0].var())
    assert ev["prediction"][0] == pytest.approx(own[:, 0].mean()) == pytest.approx(game.prediction_value)
    assert game.predicted_proba is None


def test_member_mean_at_the_empty_coalition_is_the_mixture_variance(toy_regressor, toy_background):
    """Variant A: each member becomes a mixture over the imputed points."""
    game = UncertaintyGame(toy_regressor, X0, toy_background, n_samples=50)
    out = toy_regressor(game.sampler.sample(X0, np.zeros(4, dtype=bool)))  # (M, K, 2)
    member_var = out[..., 1].mean(axis=1) + out[..., 0].var(axis=1)
    member_mean = out[..., 0].mean(axis=1)
    ev = game.evaluate(np.zeros((1, 4), dtype=bool))
    assert ev["aleatoric"][0] == pytest.approx(member_var.mean())
    assert ev["epistemic"][0] == pytest.approx(member_mean.var())


def test_each_aggregation_matches_its_definition(toy_regressor, toy_background):
    """D1 by hand for one coalition, with means and variances instead of raw moments."""
    coalition = np.array([False, True, True, False])
    games = {a: UncertaintyGame(toy_regressor, X0, toy_background, aggregation=a, n_samples=25) for a in AGGREGATIONS}
    out = toy_regressor(games["member_mean"].sampler.sample(X0, coalition))  # (M, K, 2)
    mean, var = out[..., 0], out[..., 1]
    member_mean, member_var = mean.mean(axis=1), var.mean(axis=1) + mean.var(axis=1)
    expected = {
        "member_mean": (member_var.mean(), member_mean.var()),
        "pointwise": (var.mean(), mean.var(axis=0).mean()),
        "pooled": (var.mean(), mean.var()),
    }
    totals = {}
    for aggregation, game in games.items():
        ev = game.evaluate(coalition[None, :])
        aleatoric, epistemic = expected[aggregation]
        assert ev["aleatoric"][0] == pytest.approx(aleatoric), aggregation
        assert ev["epistemic"][0] == pytest.approx(epistemic), aggregation
        assert ev["prediction"][0] == pytest.approx(mean.mean()), aggregation
        totals[aggregation] = ev["total"][0]
    # (A) and (C) describe the same mixture, so they agree on the total and differ in the split.
    assert totals["member_mean"] == pytest.approx(totals["pooled"])


def test_uncertainty_does_not_depend_on_the_target_offset(toy_regressor, toy_background):
    def shifted(X):
        out = toy_regressor(X)
        out[..., 0] += 1e7
        return out

    a = UncertaintyGame(toy_regressor, X0, toy_background, n_samples=30, measure="variance")
    b = UncertaintyGame(shifted, X0, toy_background, n_samples=30, measure="variance")
    coalitions = all_coalitions(4)
    for notion in ("total", "aleatoric", "epistemic"):
        np.testing.assert_allclose(b(coalitions, notion), a(coalitions, notion), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b(coalitions, "prediction") - 1e7, a(coalitions, "prediction"), atol=1e-6)


def test_point_predictions_only_and_task_detection(toy_regressor, toy_background, toy_model):
    means_only = UncertaintyGame(lambda X: toy_regressor(X)[..., 0], X0, toy_background, n_samples=20)
    assert means_only.task == "regression"
    assert means_only.evaluate(np.ones((1, 4), dtype=bool))["aleatoric"][0] == pytest.approx(0.0, abs=1e-12)
    # For a bare callable the task is guessed from the output: probabilities or (mean, variance).
    assert UncertaintyGame(toy_model, X0, toy_background, n_samples=20).task == "classification"
    assert UncertaintyGame(toy_regressor, X0, toy_background, n_samples=20).task == "regression"
    with pytest.raises(ValueError, match="class probabilities"):
        UncertaintyGame(toy_regressor, X0, toy_background, n_samples=20, measure="entropy")


def test_measure_must_match_the_model(data):
    X, y = data
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="regression model"):
        UncertaintyGame(model, X[0], X, measure="entropy")


def test_explanation_properties_and_roundtrip(toy_regressor, toy_background, tmp_path):
    expl = explain_uncertainty(toy_regressor, X0, toy_background, n_samples=30, measure="variance")
    assert expl.task == "regression"
    assert expl.check_additivity()
    for a in expl.attributions.values():
        assert sum(a.values.values()) == pytest.approx(a.full_value - a.baseline_value)
        for s, v in a.values.items():
            if 3 in s:
                assert v == pytest.approx(0.0, abs=1e-10)  # dummy feature
    assert "standard deviation" in expl.summary()
    expl.save(tmp_path / "e.json")
    loaded = UncertaintyExplanation.load(tmp_path / "e.json")
    assert loaded.task == "regression"
    assert loaded.predicted_proba is None
    assert loaded.summary() == expl.summary()
    for plot in (plot_uncertainty_explanation, plot_uncertainty_profile, plot_interaction_network):
        fig = plot(expl)
        fig.savefig(tmp_path / "p.png")
        plt.close(fig)


def test_regression_benchmark_generator(reg_benchmark):
    assert reg_benchmark.task == "regression"
    assert true_mean(reg_benchmark.queries["gap"])[0] == pytest.approx(0.0)
    assert true_noise_std(reg_benchmark.queries["noisy"])[0] > 5 * true_noise_std(reg_benchmark.queries["certain"])[0]
    with pytest.raises(ValueError, match="Unknown task"):
        make_uncertainty_benchmark(task="nope")


@pytest.fixture(scope="module", params=["marginal", "knn"])
def explanations(request, reg_benchmark, reg_forest):
    return {
        name: explain_uncertainty(
            reg_forest, x, reg_benchmark.X, imputer=request.param, n_samples=100, feature_names=list(FEATURE_NAMES)
        )
        for name, x in reg_benchmark.queries.items()
    }


def test_ground_truth_gap_is_an_epistemic_interaction(explanations):
    gap, certain = explanations["gap"], explanations["certain"]
    assert gap.uncertainty["epistemic"] > 5 * certain.uncertainty["epistemic"]
    name, value = gap.top_interactions("epistemic", 1)[0]
    assert name == "pair_a x pair_b"
    assert value > 0.3
    assert all(e.check_additivity(1e-7) for e in explanations.values())


def test_ground_truth_noise(explanations, reg_benchmark, reg_forest):
    assert explanations["noisy"].uncertainty["aleatoric"] > 2 * explanations["certain"].uncertainty["aleatoric"]
    # With aggregation B the noise feature is the main aleatoric driver at the noisy point.
    # (With A the aleatoric game is dominated by the variance the signal features explain.)
    pointwise = explain_uncertainty(
        reg_forest, reg_benchmark.queries["noisy"], reg_benchmark.X, aggregation="pointwise"
    )
    assert pointwise.shapley["aleatoric"].argmax() == NOISE
    assert np.abs(pointwise.shapley["aleatoric"][DUMMY]) < 0.05
