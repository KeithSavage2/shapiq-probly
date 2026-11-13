"""Out-of-distribution scores: which features make a point untypical?

The game value is an OOD score of the point when only the features in S are
known (averaged over the imputed points). The default score is model-free: the
mean distance to the ``k`` nearest reference points in standardised feature
space. Any callable ``X -> scores`` works as well, e.g. an isolation forest.

To explain OOD-ness *as seen by the model*, explain the epistemic uncertainty
with :func:`shapiq_probly.explain_uncertainty` instead.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .games import ScalarGame
from .imputation import Sampler

Score = Callable[[np.ndarray], np.ndarray]


class KNNDistanceScore:
    """Mean distance to the ``k`` nearest reference points (larger = more untypical).

    Features are standardised with the reference mean and standard deviation.
    A query point that is itself part of the reference data does not count as
    its own neighbour.

    Args:
        reference: In-distribution data, shape ``(n, d)``.
        k: Number of neighbours.
    """

    def __init__(self, reference: np.ndarray, k: int = 10) -> None:
        reference = np.asarray(reference, dtype=float)
        self.k = int(min(k, len(reference) - 1))
        if self.k < 1:
            msg = "Need at least two reference points."
            raise ValueError(msg)
        self._mean = reference.mean(axis=0)
        scale = reference.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        self._index = NearestNeighbors(n_neighbors=self.k + 1).fit(self._transform(reference))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._mean) / self._scale

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Scores of shape ``(n,)`` for points of shape ``(n, d)``."""
        dist, _ = self._index.kneighbors(self._transform(np.atleast_2d(X)))
        is_reference_point = dist[:, 0] <= 1e-12
        return np.where(is_reference_point, dist[:, 1:].mean(axis=1), dist[:, :-1].mean(axis=1))


class OODGame(ScalarGame):
    """Game whose value is the expected OOD score of the partially known point.

    Args:
        x: The point to explain, shape ``(d,)``.
        background: Background data used for imputation, shape ``(n, d)``.
        score: ``"knn"`` or a callable mapping points ``(n, d)`` to scores ``(n,)``.
        reference: In-distribution data for the kNN score (default: ``background``).
        k: Number of neighbours of the kNN score.
        imputer: ``"marginal"``, ``"gaussian"``, ``"knn"`` or a ``Sampler``.
        n_samples: Number of imputations K per coalition.
        random_state: Seed for the imputation draws.
        batch_size: Maximum number of rows per score call.
    """

    def __init__(
        self,
        x: np.ndarray,
        background: np.ndarray,
        *,
        score: str | Score = "knn",
        reference: np.ndarray | None = None,
        k: int = 10,
        imputer: str | Sampler = "marginal",
        n_samples: int = 100,
        random_state: int | None = 0,
        batch_size: int = 50_000,
    ) -> None:
        super().__init__(
            x,
            background,
            name="OOD score",
            imputer=imputer,
            n_samples=n_samples,
            random_state=random_state,
            batch_size=batch_size,
        )
        if callable(score):
            self.score = score
            score_name = getattr(score, "__name__", type(score).__name__)
        elif score == "knn":
            self.score = KNNDistanceScore(background if reference is None else reference, k)
            score_name = f"knn (k={self.score.k})"
        else:
            msg = f"Unknown score {score!r}. Use 'knn' or pass a callable."
            raise ValueError(msg)
        self.info = {"score": score_name}

    def _chunk_values(self, points: list[np.ndarray]) -> list[np.ndarray]:
        sizes = np.cumsum([len(p) for p in points])[:-1]
        scores = np.asarray(self.score(np.concatenate(points)), dtype=float)
        return [np.array([float(s.mean())]) for s in np.split(scores, sizes)]
