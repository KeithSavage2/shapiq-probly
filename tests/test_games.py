import numpy as np
import pytest

from shapiq_probly import UncertaintyGame, all_coalitions, entropy_decomposition
from shapiq_probly.games import AGGREGATIONS


@pytest.fixture
def x():
    return np.array([0.8, -0.3, 1.2, 0.5])


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
@pytest.mark.parametrize("imputer", ["marginal", "gaussian", "knn"])
def test_additivity_for_every_coalition(toy_model, toy_background, x, aggregation, imputer):
    game = UncertaintyGame(toy_model, x, toy_background, aggregation=aggregation, imputer=imputer, n_samples=40)
    ev = game.evaluate(all_coalitions(4))
    np.testing.assert_allclose(ev["total"], ev["aleatoric"] + ev["epistemic"], atol=1e-12)
    assert (ev["epistemic"] >= -1e-12).all()


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_full_coalition_is_the_models_own_uncertainty(toy_model, toy_background, x, aggregation):
    game = UncertaintyGame(toy_model, x, toy_background, aggregation=aggregation, n_samples=40)
    ev = game.evaluate(np.ones((1, 4), dtype=bool))
    tu, au, eu = entropy_decomposition(toy_model(x[None, :])[:, 0, :])
    assert ev["total"][0] == pytest.approx(tu)
    assert ev["aleatoric"][0] == pytest.approx(au)
    assert ev["epistemic"][0] == pytest.approx(eu)
    assert ev["prediction"][0] == pytest.approx(game.predicted_proba[game.predicted_class])


def test_d1_variants_a_and_c_share_total_but_split_differently(toy_model, toy_background, x):
    """Imputation spread ends up in AU for (A) and partly in EU for (C)."""
    kwargs = {"n_samples": 60, "random_state": 0}
    a = UncertaintyGame(toy_model, x, toy_background, aggregation="member_mean", **kwargs).evaluate(all_coalitions(4))
    c = UncertaintyGame(toy_model, x, toy_background, aggregation="pooled", **kwargs).evaluate(all_coalitions(4))
    np.testing.assert_allclose(a["total"], c["total"], atol=1e-12)
    assert (c["epistemic"] >= a["epistemic"] - 1e-12).all()
    assert c["epistemic"][0] > a["epistemic"][0] + 1e-3


def test_information_gain_mode(toy_model, toy_background, x):
    u = UncertaintyGame(toy_model, x, toy_background, mode="uncertainty", n_samples=40)
    g = UncertaintyGame(toy_model, x, toy_background, mode="information_gain", n_samples=40)
    coalitions = all_coalitions(4)
    for notion in ("total", "aleatoric", "epistemic"):
        np.testing.assert_allclose(g(coalitions, notion), u(coalitions, notion)[0] - u(coalitions, notion), atol=1e-12)
        assert g(coalitions, notion)[0] == 0.0
    np.testing.assert_allclose(g(coalitions, "prediction"), u(coalitions, "prediction"))


def test_cache_and_determinism(toy_model, toy_background, x):
    calls = []

    def counting(X):
        calls.append(len(X))
        return toy_model(X)

    game = UncertaintyGame(counting, x, toy_background, n_samples=20)
    coalitions = all_coalitions(4)
    first = game.evaluate(coalitions)
    n_calls = len(calls)
    second = game.evaluate(coalitions[::-1])
    assert len(calls) == n_calls
    assert game.n_evaluated == 16
    np.testing.assert_array_equal(first["total"], second["total"][::-1])
    again = UncertaintyGame(toy_model, x, toy_background, n_samples=20).evaluate(coalitions)
    np.testing.assert_array_equal(first["epistemic"], again["epistemic"])


def test_batching_gives_same_values(toy_model, toy_background, x):
    big = UncertaintyGame(toy_model, x, toy_background, n_samples=20).evaluate(all_coalitions(4))
    small = UncertaintyGame(toy_model, x, toy_background, n_samples=20, batch_size=30).evaluate(all_coalitions(4))
    np.testing.assert_allclose(big["total"], small["total"], atol=1e-12)


def test_input_validation(toy_model, toy_background, x):
    with pytest.raises(ValueError, match="Unknown aggregation"):
        UncertaintyGame(toy_model, x, toy_background, aggregation="nope")
    with pytest.raises(ValueError, match="Unknown mode"):
        UncertaintyGame(toy_model, x, toy_background, mode="nope")
    with pytest.raises(ValueError, match="features"):
        UncertaintyGame(toy_model, x[:3], toy_background)
    with pytest.raises(ValueError, match=r"\(M, n, C\)"):
        UncertaintyGame(lambda X: np.ones((len(X), 2)), x, toy_background)
