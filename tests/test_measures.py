import numpy as np
import pytest
from scipy.optimize import minimize

from shapiq_probly.measures import (
    credal_decomposition,
    entropy,
    entropy_decomposition,
    get_measure,
    to_moments,
    upper_entropy,
    variance_decomposition,
)


def test_entropy_known_values():
    assert entropy(np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert entropy(np.array([1.0, 0.0])) == 0.0


def test_decomposition_is_additive_and_nonnegative():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=(7, 20))
    tu, au, eu = entropy_decomposition(probs)
    assert tu.shape == (20,)
    np.testing.assert_allclose(tu, au + eu, atol=1e-12)
    assert (eu >= -1e-12).all()


def test_disagreeing_members_are_purely_epistemic():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    tu, au, eu = entropy_decomposition(probs)
    assert au == pytest.approx(0.0)
    assert tu == pytest.approx(np.log(2)) == pytest.approx(eu)


def test_agreeing_members_are_purely_aleatoric():
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    _, au, eu = entropy_decomposition(probs)
    assert eu == pytest.approx(0.0)
    assert au == pytest.approx(np.log(2))


def test_unknown_measure():
    with pytest.raises(ValueError, match="Unknown measure"):
        get_measure("nope")


def test_variance_decomposition_is_the_law_of_total_variance():
    rng = np.random.default_rng(0)
    means, variances = rng.normal(size=(6, 15)), rng.uniform(0.1, 2.0, size=(6, 15))
    tu, au, eu = variance_decomposition(to_moments(means, variances))
    np.testing.assert_allclose(au, variances.mean(axis=0))
    np.testing.assert_allclose(eu, means.var(axis=0))
    np.testing.assert_allclose(tu, au + eu, atol=1e-12)


def test_averaging_moments_is_mixing_distributions():
    """The variance of an equal mixture of Gaussians, checked by sampling."""
    rng = np.random.default_rng(0)
    means, variances = np.array([-1.0, 0.5, 3.0]), np.array([0.2, 1.0, 0.5])
    first, second = to_moments(means, variances).mean(axis=0)
    samples = rng.normal(means, np.sqrt(variances), size=(400_000, 3)).ravel()
    assert first == pytest.approx(samples.mean(), abs=0.01)
    assert second - first**2 == pytest.approx(samples.var(), rel=0.01)


def test_upper_entropy_binary_closed_form():
    straddling = np.array([[0.9, 0.1], [0.2, 0.8]])
    one_sided = np.array([[0.9, 0.1], [0.7, 0.3]])
    assert upper_entropy(straddling) == pytest.approx(np.log(2))
    assert upper_entropy(one_sided) == pytest.approx(entropy(np.array([0.7, 0.3])))


def test_upper_entropy_multiclass_matches_a_generic_solver():
    rng = np.random.default_rng(0)
    for _ in range(15):
        n_members, n_classes = rng.integers(2, 8), rng.integers(3, 6)
        probs = rng.dirichlet(np.full(n_classes, rng.choice([0.3, 1.0, 5.0])), size=n_members)
        reference = -minimize(
            lambda w, p=probs: -entropy(w @ p),
            np.full(n_members, 1 / n_members),
            bounds=[(0, 1)] * n_members,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 500},
        ).fun
        ours = float(upper_entropy(probs))
        assert ours == pytest.approx(reference, abs=1e-6)
        assert ours >= entropy(probs.mean(axis=0)) - 1e-9


def test_upper_entropy_is_maximal_if_the_uniform_distribution_is_in_the_hull():
    probs = np.eye(4)[:, None, :].repeat(3, axis=1)  # four members, batch of three
    np.testing.assert_allclose(upper_entropy(probs), np.log(4), atol=1e-8)


def test_credal_decomposition():
    rng = np.random.default_rng(1)
    for n_classes in (2, 4):
        probs = rng.dirichlet(np.ones(n_classes), size=(6, 25))
        tu, au, eu = credal_decomposition(probs)
        np.testing.assert_allclose(tu, au + eu, atol=1e-12)
        np.testing.assert_allclose(au, entropy(probs).min(axis=0))
        # The credal set contains the mean, so it is at least as wide as the mutual information.
        assert (eu >= entropy_decomposition(probs)[2] - 1e-9).all()
    agree = np.tile([0.7, 0.2, 0.1], (5, 1))
    assert credal_decomposition(agree)[2] == pytest.approx(0.0, abs=1e-9)


def test_measures_know_their_task():
    assert get_measure("entropy").task == "classification" == get_measure("credal").task
    assert get_measure("variance").task == "regression"
    assert get_measure(entropy_decomposition).name == "entropy_decomposition"
