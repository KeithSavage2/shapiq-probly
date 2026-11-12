"""The uncertainty game: v(S) = uncertainty if only the features in S are known.

One imputation pass yields four games at once (total, aleatoric, epistemic and
the probability of the predicted class). Sharing the pass saves a factor of
three and makes ``v_TU(S) = v_AU(S) + v_EU(S)`` hold for every coalition.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import as_member_predictor
from .imputation import Sampler, get_sampler
from .measures import NOTIONS, Decomposition, get_decomposition

GAMES = (*NOTIONS, "prediction")
AGGREGATIONS = ("member_mean", "pointwise", "pooled")
MODES = ("uncertainty", "information_gain")


class UncertaintyGame:
    """Cooperative game over features whose value is a model's uncertainty.

    Args:
        model: Member predictor ``X -> (M, n, C)``, sklearn ensemble or list of
            classifiers (see :func:`shapiq_probly.adapters.as_member_predictor`).
        x: The point to explain, shape ``(d,)``.
        background: Background data used for imputation, shape ``(n, d)``.
        imputer: ``"marginal"``, ``"gaussian"``, ``"knn"`` or a ``Sampler``.
        aggregation: Order of averaging and measuring (design question D1).
            ``"member_mean"`` (A): average each member over the imputations,
            then decompose. ``"pointwise"`` (B): decompose at every imputed
            point, then average. ``"pooled"`` (C): treat all member x
            imputation predictions as one large ensemble.
        mode: ``"uncertainty"`` gives ``v(S) = U(S)``; ``"information_gain"``
            gives ``v(S) = U(empty) - U(S)`` (design question D3). The
            prediction game is not affected.
        measure: Name of a decomposition or a callable ``(M, ..., C) -> (TU, AU, EU)``.
        n_samples: Number of imputations K per coalition.
        random_state: Seed for the imputation draws.
        batch_size: Maximum number of rows per model call.
    """

    def __init__(
        self,
        model: Any,
        x: np.ndarray,
        background: np.ndarray,
        *,
        imputer: str | Sampler = "marginal",
        aggregation: str = "member_mean",
        mode: str = "uncertainty",
        measure: str | Decomposition = "entropy",
        n_samples: int = 100,
        random_state: int | None = 0,
        batch_size: int = 50_000,
    ) -> None:
        if aggregation not in AGGREGATIONS:
            msg = f"Unknown aggregation {aggregation!r}. Available: {AGGREGATIONS}."
            raise ValueError(msg)
        if mode not in MODES:
            msg = f"Unknown mode {mode!r}. Available: {MODES}."
            raise ValueError(msg)
        self.predict_members = as_member_predictor(model)
        self.x = np.asarray(x, dtype=float).ravel()
        self.sampler = get_sampler(imputer, background, n_samples, random_state)
        if self.x.shape[0] != self.sampler.n_features:
            msg = f"x has {self.x.shape[0]} features, background has {self.sampler.n_features}."
            raise ValueError(msg)
        self.n_players = self.x.shape[0]
        self.aggregation = aggregation
        self.mode = mode
        self.decompose = get_decomposition(measure)
        self.batch_size = batch_size
        self._cache: dict[bytes, np.ndarray] = {}

        probs = self._check(self.predict_members(self.x[None, :]))[:, 0, :]
        #: Mean predictive distribution at ``x`` and the class it favours.
        self.predicted_proba = probs.mean(axis=0)
        self.predicted_class = int(self.predicted_proba.argmax())
        self.n_members = probs.shape[0]
        self._empty = self._raw(np.zeros((1, self.n_players), dtype=bool))[0]

    @staticmethod
    def _check(probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 3:
            msg = f"The member predictor must return shape (M, n, C), got {probs.shape}."
            raise ValueError(msg)
        return probs

    def _raw_one(self, probs: np.ndarray) -> np.ndarray:
        """Raw values (TU, AU, EU, prediction) from predictions of shape (M, K, C)."""
        if self.aggregation == "member_mean":
            tu, au, eu = self.decompose(probs.mean(axis=1))
        elif self.aggregation == "pointwise":
            tu, au, eu = (np.mean(v) for v in self.decompose(probs))
        else:  # pooled
            tu, au, eu = self.decompose(probs.reshape(-1, probs.shape[-1]))
        return np.array([tu, au, eu, probs[..., self.predicted_class].mean()], dtype=float)

    def _raw(self, coalitions: np.ndarray) -> np.ndarray:
        """Raw values of shape ``(n_coalitions, 4)``, cached per coalition."""
        out = np.empty((len(coalitions), len(GAMES)))
        keys = [c.tobytes() for c in coalitions]
        pending, seen = [], set(self._cache)
        for i, key in enumerate(keys):
            if key not in seen:
                seen.add(key)
                pending.append(i)
        per_batch = max(1, self.batch_size // max(1, self.sampler.n_samples))
        for start in range(0, len(pending), per_batch):
            chunk = pending[start : start + per_batch]
            points = [self.sampler.sample(self.x, coalitions[i]) for i in chunk]
            sizes = np.cumsum([len(p) for p in points])[:-1]
            probs = self._check(self.predict_members(np.concatenate(points)))
            for i, p in zip(chunk, np.split(probs, sizes, axis=1), strict=True):
                self._cache[keys[i]] = self._raw_one(p)
        for i, k in enumerate(keys):
            out[i] = self._cache[k]
        return out

    def evaluate(self, coalitions: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate all four games on a boolean coalition matrix ``(n, d)``.

        Returns:
            Mapping from ``"total"``, ``"aleatoric"``, ``"epistemic"`` and
            ``"prediction"`` to arrays of shape ``(n,)``.
        """
        coalitions = np.atleast_2d(np.asarray(coalitions, dtype=bool))
        raw = self._raw(coalitions)
        if self.mode == "information_gain":
            raw = raw.copy()
            raw[:, :3] = self._empty[:3] - raw[:, :3]
        return {name: raw[:, j] for j, name in enumerate(GAMES)}

    def value_function(self, coalitions: np.ndarray, notion: str = "total") -> np.ndarray:
        """Values of one game, in the form expected by ``shapiq.Game``."""
        if notion not in GAMES:
            msg = f"Unknown notion {notion!r}. Available: {GAMES}."
            raise ValueError(msg)
        return self.evaluate(coalitions)[notion]

    def __call__(self, coalitions: np.ndarray, notion: str = "total") -> np.ndarray:
        return self.value_function(coalitions, notion)

    @property
    def n_evaluated(self) -> int:
        """Number of distinct coalitions evaluated so far."""
        return len(self._cache)
