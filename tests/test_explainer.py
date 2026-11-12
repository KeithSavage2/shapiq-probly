import warnings

import numpy as np
import pytest

from shapiq_probly import UncertaintyExplainer, UncertaintyExplanation, explain_uncertainty

X0 = np.array([0.8, 0.8, 1.2, 0.5])  # features 0 and 1 have the same value


@pytest.mark.parametrize("index", ["SV", "k-SII", "SII", "Moebius"])
@pytest.mark.parametrize("aggregation", ["member_mean", "pointwise", "pooled"])
def test_explanations_are_additive(toy_model, toy_background, index, aggregation):
    expl = explain_uncertainty(toy_model, X0, toy_background, index=index, aggregation=aggregation, n_samples=30)
    assert expl.check_additivity()
    for s, total in expl.total.values.items():
        assert total == pytest.approx(expl.aleatoric.values[s] + expl.epistemic.values[s])


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic", "prediction"])
def test_efficiency(toy_model, toy_background, notion):
    expl = explain_uncertainty(toy_model, X0, toy_background, n_samples=30)
    a = expl.attributions[notion]
    assert sum(a.values.values()) == pytest.approx(a.full_value - a.baseline_value)
    assert expl.shapley[notion].sum() == pytest.approx(a.full_value - a.baseline_value)


def test_dummy_feature_gets_zero(toy_model, toy_background):
    expl = explain_uncertainty(toy_model, X0, toy_background, n_samples=30)
    for a in expl.attributions.values():
        for s, v in a.values.items():
            if 3 in s:
                assert v == pytest.approx(0.0, abs=1e-12)


def test_symmetric_features_get_equal_values(toy_model, toy_background):
    expl = explain_uncertainty(toy_model, X0, toy_background, n_samples=30)
    for notion, sv in expl.shapley.items():
        assert sv[0] == pytest.approx(sv[1]), notion
    assert expl.epistemic[(0, 2)] == pytest.approx(expl.epistemic[(1, 2)])


def test_information_gain_flips_the_sign(toy_model, toy_background):
    u = explain_uncertainty(toy_model, X0, toy_background, n_samples=30)
    g = explain_uncertainty(toy_model, X0, toy_background, n_samples=30, mode="information_gain")
    np.testing.assert_allclose(g.shapley["epistemic"], -u.shapley["epistemic"], atol=1e-12)
    np.testing.assert_allclose(g.shapley["prediction"], u.shapley["prediction"], atol=1e-12)
    assert g.uncertainty == pytest.approx(u.uncertainty)


def test_uncertainty_at_x_and_summary(toy_model, toy_background):
    expl = explain_uncertainty(toy_model, X0, toy_background, n_samples=30, feature_names=list("abcd"))
    u = expl.uncertainty
    assert u["total"] == pytest.approx(u["aleatoric"] + u["epistemic"])
    assert expl.total.full_value == pytest.approx(u["total"])
    text = expl.summary()
    assert "Prediction: class" in text
    assert "epistemic" in text
    assert len(expl.top_interactions("epistemic", 3)) == 3
    assert set(expl.moebius("epistemic")) == set(expl.epistemic.values)


def test_json_roundtrip(toy_model, toy_background, tmp_path):
    expl = explain_uncertainty(toy_model, X0, toy_background, n_samples=30)
    path = tmp_path / "expl.json"
    expl.save(path)
    loaded = UncertaintyExplanation.load(path)
    assert loaded.epistemic.values == pytest.approx(expl.epistemic.values)
    np.testing.assert_allclose(loaded.shapley["total"], expl.shapley["total"])
    np.testing.assert_allclose(loaded.game_values["aleatoric"], expl.game_values["aleatoric"])
    assert loaded.summary() == expl.summary()
    assert loaded.check_additivity()


def test_approximation_fallback_is_close_and_still_additive(toy_model, toy_background):
    exact = explain_uncertainty(toy_model, X0, toy_background, n_samples=30, index="SV")
    with pytest.warns(UserWarning, match="approximating"):
        approx = explain_uncertainty(toy_model, X0, toy_background, n_samples=30, max_exact_features=3, budget=800)
    assert approx.total.estimated
    assert approx.game_values is None
    assert approx.check_additivity()
    for notion in exact.shapley:
        np.testing.assert_allclose(approx.shapley[notion], exact.shapley[notion], atol=0.03)
    with pytest.raises(ValueError, match="exact game values"):
        approx.moebius()


def test_explain_many_and_feature_name_check(toy_model, toy_background):
    explainer = UncertaintyExplainer(toy_model, toy_background, n_samples=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(explainer.explain_many(toy_background[:3])) == 3
    with pytest.raises(ValueError, match="feature names"):
        UncertaintyExplainer(toy_model, toy_background, feature_names=["a"])
