"""Shared fixtures."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from shapiq_probly import make_uncertainty_benchmark

mpl.use("Agg")


@pytest.fixture(scope="session")
def benchmark():
    return make_uncertainty_benchmark(n_samples=3000, random_state=0)


@pytest.fixture(scope="session")
def forest(benchmark):
    model = RandomForestClassifier(n_estimators=40, min_samples_leaf=10, random_state=0)
    return model.fit(benchmark.X, benchmark.y)


@pytest.fixture
def toy_model():
    """Three 'members' that ignore feature 3 and treat features 0 and 1 symmetrically."""

    def predict(X: np.ndarray) -> np.ndarray:
        s = X[:, 0] + X[:, 1]
        logits = np.stack([1.5 * s + 0.5 * X[:, 2], 0.5 * s - X[:, 2], 2.0 * s * X[:, 2]])
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1 - p, p], axis=-1)

    return predict


@pytest.fixture
def toy_background():
    rng = np.random.default_rng(1)
    bg = rng.normal(size=(300, 4))
    bg[:, 1] = bg[:, 0]  # features 0 and 1 are exchangeable in the background
    return bg


@pytest.fixture(scope="session")
def calibration():
    """Fresh data the forest has not seen, for conformal calibration."""
    data = make_uncertainty_benchmark(n_samples=1500, random_state=1)
    return data.X, data.y


@pytest.fixture(scope="session")
def reg_benchmark():
    return make_uncertainty_benchmark(n_samples=3000, random_state=0, task="regression")


@pytest.fixture(scope="session")
def reg_forest(reg_benchmark):
    model = RandomForestRegressor(n_estimators=40, min_samples_leaf=10, random_state=0)
    return model.fit(reg_benchmark.X, reg_benchmark.y)


@pytest.fixture
def toy_regressor():
    """Three regression 'members' (mean and variance) that ignore feature 3."""

    def predict(X: np.ndarray) -> np.ndarray:
        s = X[:, 0] + X[:, 1]
        means = np.stack([s + 0.5 * X[:, 2], 1.2 * s - 0.3 * X[:, 2], s * X[:, 2]])
        variances = np.stack([0.1 + X[:, 2] ** 2, 0.2 + 0.0 * s, 0.05 + 0.1 * s**2])
        return np.stack([means, variances], axis=-1)

    return predict
