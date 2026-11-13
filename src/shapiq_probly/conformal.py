"""Conformal prediction: which features inflate the prediction set? (classification).

Split conformal prediction with the threshold ("LAC") score
``s(x, y) = 1 - p(y | x)`` on the mean ensemble probability. The prediction set
contains every class with ``p(y | x) >= 1 - q_hat``, where ``q_hat`` is the
conformal quantile of the calibration scores. The game value is the size of
that set when only the features in S are known.

There is no aleatoric/epistemic split here: the set size is a single number.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import as_member_predictor
from .games import ScalarGame
from .imputation import Sampler

SET_AGGREGATIONS = ("pointwise", "pooled")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: the ``ceil((n+1)(1-alpha))``-th smallest score.

    Returns infinity if there are too few calibration points for the requested
    coverage (the prediction set is then every class).
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha) - 1e-12))
    if rank > n:
        return float("inf")
    return float(scores[rank - 1])


class ConformalSetSizeGame(ScalarGame):
    """Game whose value is the size of the conformal prediction set.

    Args:
        model: Member predictor ``X -> (M, n, C)``, sklearn ensemble or list of
            classifiers.
        x: The point to explain, shape ``(d,)``.
        background: Background data used for imputation, shape ``(n, d)``.
        calibration: ``(X_cal, y_cal)``, data the model was **not** trained on.
        alpha: Miscoverage level; sets cover the true class with probability
            of at least ``1 - alpha``.
        aggregation: ``"pointwise"`` (default): expected set size over the
            imputed points, so ``v(empty)`` is the average set size on the
            background data. ``"pooled"``: size of the set built from the
            averaged predictive distribution (integer-valued).
        classes: Class labels in the order of the probability columns.
            Defaults to ``model.classes_``; without it ``y_cal`` must already
            hold column indices.
        imputer: ``"marginal"``, ``"gaussian"``, ``"knn"`` or a ``Sampler``.
        n_samples: Number of imputations K per coalition.
        random_state: Seed for the imputation draws.
        batch_size: Maximum number of rows per model call.
    """

    def __init__(
        self,
        model: Any,
        x: np.ndarray,
        background: np.ndarray,
        calibration: tuple[np.ndarray, np.ndarray],
        *,
        alpha: float = 0.1,
        aggregation: str = "pointwise",
        classes: np.ndarray | None = None,
        imputer: str | Sampler = "marginal",
        n_samples: int = 100,
        random_state: int | None = 0,
        batch_size: int = 50_000,
    ) -> None:
        if aggregation not in SET_AGGREGATIONS:
            msg = f"Unknown aggregation {aggregation!r}. Available: {SET_AGGREGATIONS}."
            raise ValueError(msg)
        if not 0 < alpha < 1:
            msg = "alpha must be in (0, 1)."
            raise ValueError(msg)
        super().__init__(
            x,
            background,
            name="prediction set size",
            imputer=imputer,
            n_samples=n_samples,
            random_state=random_state,
            batch_size=batch_size,
        )
        self.predict_members = as_member_predictor(model)
        self.aggregation = aggregation
        self.alpha = alpha

        X_cal, y_cal = np.asarray(calibration[0], dtype=float), np.asarray(calibration[1])
        proba = self._proba(X_cal)
        if classes is None:
            classes = getattr(model, "classes_", None)
        self.classes = np.arange(proba.shape[1]) if classes is None else np.asarray(classes)
        columns = np.searchsorted(self.classes, y_cal)
        if (columns >= len(self.classes)).any() or (self.classes[columns] != y_cal).any():
            msg = "y_cal contains labels that are not in 'classes'."
            raise ValueError(msg)
        self.qhat = conformal_quantile(1.0 - proba[np.arange(len(y_cal)), columns], alpha)
        #: A class enters the set if its probability is at least this value.
        self.threshold = 1.0 - self.qhat
        own = self._proba(self.x[None, :])[0]
        #: The conformal prediction set at ``x``.
        self.prediction_set = self.classes[own >= self.threshold]
        self.info = {
            "alpha": alpha,
            "threshold": float(self.threshold),
            "prediction_set": self.prediction_set.tolist(),
            "aggregation": aggregation,
            "n_calibration": len(y_cal),
        }

    def _proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.asarray(self.predict_members(X), dtype=float)
        if probs.ndim != 3:
            msg = f"The member predictor must return shape (M, n, C), got {probs.shape}."
            raise ValueError(msg)
        return probs.mean(axis=0)

    def set_sizes(self, X: np.ndarray) -> np.ndarray:
        """Prediction-set size for every row of ``X``."""
        return (self._proba(np.atleast_2d(X)) >= self.threshold).sum(axis=1)

    def _chunk_values(self, points: list[np.ndarray]) -> list[np.ndarray]:
        sizes = np.cumsum([len(p) for p in points])[:-1]
        out = []
        for proba in np.split(self._proba(np.concatenate(points)), sizes):
            if self.aggregation == "pooled":
                value = (proba.mean(axis=0) >= self.threshold).sum()
            else:
                value = (proba >= self.threshold).sum(axis=1).mean()
            out.append(np.array([float(value)]))
        return out
