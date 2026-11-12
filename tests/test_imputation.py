import numpy as np
import pytest

from shapiq_probly.imputation import GaussianSampler, KNNSampler, MarginalSampler, get_sampler


@pytest.fixture
def background():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    return np.column_stack([a, a + 0.1 * rng.normal(size=2000), rng.normal(size=2000)])


@pytest.mark.parametrize("name", ["marginal", "gaussian", "knn"])
def test_known_features_are_kept_and_draws_are_reused(background, name):
    sampler = get_sampler(name, background, n_samples=50, random_state=0)
    x = np.array([2.0, 2.0, -1.0])
    s = np.array([True, False, True])
    out = sampler.sample(x, s)
    assert out.shape == (50, 3)
    np.testing.assert_array_equal(out[:, s], np.tile(x[s], (50, 1)))
    np.testing.assert_array_equal(out, sampler.sample(x, s))
    np.testing.assert_array_equal(sampler.sample(x, np.ones(3, dtype=bool)), np.tile(x, (50, 1)))


def test_marginal_ignores_correlation_conditional_respects_it(background):
    x = np.array([2.0, 2.0, 0.0])
    s = np.array([True, False, False])
    marginal = MarginalSampler(background, 200).sample(x, s)[:, 1].mean()
    gaussian = GaussianSampler(background, 200).sample(x, s)[:, 1].mean()
    knn = KNNSampler(background, 50).sample(x, s)[:, 1].mean()
    assert abs(marginal) < 0.3
    assert gaussian == pytest.approx(2.0, abs=0.15)
    assert knn == pytest.approx(2.0, abs=0.3)


def test_unknown_imputer(background):
    with pytest.raises(ValueError, match="Unknown imputer"):
        get_sampler("nope", background)
