"""Synthetic benchmark with built-in ground truth for uncertainty explanations.

Four features, binary label:

* ``noise``  controls label noise. Above ``noise_threshold`` the label is a
  coin flip (aleatoric uncertainty), below it is almost deterministic.
* ``pair_a``, ``pair_b`` carry the signal (``pair_a + pair_b``). The corner
  ``pair_a > gap, pair_b < -gap`` never occurs in the data, although each value
  is perfectly normal on its own. A model cannot know what happens there:
  epistemic uncertainty that only exists as an *interaction* (N1).
* ``dummy`` is independent of everything (expected attribution: about 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FEATURE_NAMES = ("noise", "pair_a", "pair_b", "dummy")


@dataclass
class UncertaintyBenchmark:
    """Data and ground truth of the synthetic benchmark."""

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    #: Query points with a known expected explanation.
    queries: dict[str, np.ndarray] = field(default_factory=dict)
    #: Which features are expected to drive which kind of uncertainty.
    ground_truth: dict[str, tuple[str, ...]] = field(default_factory=dict)


def in_gap(X: np.ndarray, gap: float = 0.2) -> np.ndarray:
    """Whether points lie in the unobserved corner of the feature pair."""
    X = np.atleast_2d(X)
    return (X[:, 1] > gap) & (X[:, 2] < -gap)


def true_proba(X: np.ndarray, steepness: float = 6.0, noise_threshold: float = 0.8) -> np.ndarray:
    """Ground-truth probability of class 1."""
    X = np.atleast_2d(X)
    signal = 1.0 / (1.0 + np.exp(-steepness * (X[:, 1] + X[:, 2])))
    noisy = 1.0 / (1.0 + np.exp(-40.0 * (X[:, 0] - noise_threshold)))
    return 0.5 + (signal - 0.5) * (1.0 - noisy)


def make_uncertainty_benchmark(
    n_samples: int = 4000,
    gap: float = 0.2,
    random_state: int | None = 0,
) -> UncertaintyBenchmark:
    """Generate the benchmark.

    Args:
        n_samples: Number of samples after removing the gap region.
        gap: The corner ``pair_a > gap`` and ``pair_b < -gap`` is left empty.
        random_state: Seed.
    """
    rng = np.random.default_rng(random_state)
    X = np.empty((0, 4))
    while len(X) < n_samples:
        cand = np.column_stack(
            [
                rng.uniform(0, 1, n_samples),
                rng.uniform(-1, 1, n_samples),
                rng.uniform(-1, 1, n_samples),
                rng.uniform(-1, 1, n_samples),
            ]
        )
        X = np.vstack([X, cand[~in_gap(cand, gap)]])
    X = X[:n_samples]
    y = (rng.uniform(size=n_samples) < true_proba(X)).astype(int)
    queries = {
        # low noise, both pair values individually common, combination unseen
        "gap": np.array([0.1, 0.7, -0.7, 0.0]),
        # high noise, clear signal region
        "noisy": np.array([0.95, 0.6, 0.6, 0.0]),
        # low noise, clear signal region: the model should be certain
        "certain": np.array([0.05, 0.6, 0.6, 0.0]),
    }
    ground_truth = {
        "aleatoric": ("noise",),
        "epistemic_interaction": ("pair_a", "pair_b"),
        "dummy": ("dummy",),
    }
    return UncertaintyBenchmark(X=X, y=y, queries=queries, ground_truth=ground_truth)
