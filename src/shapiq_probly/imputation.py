"""Feature removal: draw the unknown features of ``x`` given a coalition.

Unlike the shapiq imputers, the samplers here only produce *imputed inputs*.
Aggregation of the model outputs happens later in the game, because for
uncertainty we must average distributions first and measure afterwards.

All samplers use common random numbers: the randomness is drawn once at
construction and reused for every coalition. Differences between coalition
values (and therefore Shapley and Moebius values) are much less noisy this way.
"""

from __future__ import annotations

import numpy as np


class Sampler:
    """Base class. ``sample(x, coalition)`` returns imputed points ``(K, d)``."""

    def __init__(self, background: np.ndarray, n_samples: int = 100, random_state: int | None = 0) -> None:
        background = np.asarray(background, dtype=float)
        if background.ndim != 2:
            msg = "background must be a 2D array of shape (n_background, n_features)."
            raise ValueError(msg)
        self.background = background
        self.n_features = background.shape[1]
        self.n_samples = int(min(n_samples, len(background)))
        self.rng = np.random.default_rng(random_state)

    def sample(self, x: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        """Return ``(K, d)`` points that equal ``x`` on the coalition."""
        raise NotImplementedError


class MarginalSampler(Sampler):
    """Replace unknown features by rows of the background data (interventional).

    The same ``n_samples`` background rows are used for every coalition. Note
    that this creates points off the data manifold (design question D2).
    """

    def __init__(self, background: np.ndarray, n_samples: int = 100, random_state: int | None = 0) -> None:
        super().__init__(background, n_samples, random_state)
        rows = self.rng.choice(len(self.background), size=self.n_samples, replace=False)
        self._draws = self.background[rows]

    def sample(self, x: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        out = self._draws.copy()
        out[:, coalition] = x[coalition]
        return out


class GaussianSampler(Sampler):
    """Sample unknown features from a multivariate Gaussian conditioned on ``x_S``.

    Only meaningful for numerical features. The standard-normal draws are fixed
    at construction (common random numbers).
    """

    def __init__(self, background: np.ndarray, n_samples: int = 100, random_state: int | None = 0) -> None:
        super().__init__(background, n_samples, random_state)
        self.n_samples = int(n_samples)
        self._mean = self.background.mean(axis=0)
        cov = np.atleast_2d(np.cov(self.background, rowvar=False))
        self._cov = cov + 1e-8 * np.eye(self.n_features)
        self._z = self.rng.standard_normal((self.n_samples, self.n_features))

    def sample(self, x: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        known, unknown = np.flatnonzero(coalition), np.flatnonzero(~coalition)
        out = np.tile(x, (self.n_samples, 1))
        if unknown.size == 0:
            return out
        mean, cov = self._mean[unknown], self._cov[np.ix_(unknown, unknown)]
        if known.size:
            cross = self._cov[np.ix_(unknown, known)]
            solve = np.linalg.solve(self._cov[np.ix_(known, known)], cross.T).T
            mean = mean + solve @ (x[known] - self._mean[known])
            cov = cov - solve @ cross.T
        cov = (cov + cov.T) / 2 + 1e-10 * np.eye(unknown.size)
        chol = np.linalg.cholesky(cov)
        out[:, unknown] = mean + self._z[:, unknown] @ chol.T
        return out


class KNNSampler(Sampler):
    """Empirical conditional: take unknown features from the nearest neighbours.

    For a coalition S, the ``n_samples`` background rows closest to ``x`` in the
    (standardised) known features provide the values of the unknown features.
    Imputed points therefore stay close to the data manifold. For the empty
    coalition this reduces to the marginal sampler.
    """

    def __init__(self, background: np.ndarray, n_samples: int = 100, random_state: int | None = 0) -> None:
        super().__init__(background, n_samples, random_state)
        scale = self.background.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        rows = self.rng.choice(len(self.background), size=self.n_samples, replace=False)
        self._marginal_rows = rows

    def sample(self, x: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        if coalition.any():
            diff = (self.background[:, coalition] - x[coalition]) / self._scale[coalition]
            dist = np.einsum("ij,ij->i", diff, diff)
            rows = np.argsort(dist, kind="stable")[: self.n_samples]
        else:
            rows = self._marginal_rows
        out = self.background[rows].copy()
        out[:, coalition] = x[coalition]
        return out


SAMPLERS: dict[str, type[Sampler]] = {
    "marginal": MarginalSampler,
    "gaussian": GaussianSampler,
    "knn": KNNSampler,
}


def get_sampler(
    imputer: str | Sampler,
    background: np.ndarray,
    n_samples: int = 100,
    random_state: int | None = 0,
) -> Sampler:
    """Resolve a sampler given by name or as an instance."""
    if isinstance(imputer, Sampler):
        return imputer
    try:
        cls = SAMPLERS[imputer]
    except KeyError:
        msg = f"Unknown imputer {imputer!r}. Available: {sorted(SAMPLERS)}."
        raise ValueError(msg) from None
    return cls(background, n_samples=n_samples, random_state=random_state)
