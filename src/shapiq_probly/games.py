"""Games over features: v(S) = a quantity if only the features in S are known.

:class:`UncertaintyGame` yields four games from one imputation pass (total,
aleatoric, epistemic uncertainty and the prediction). Sharing the pass saves a
factor of three and makes ``v_TU(S) = v_AU(S) + v_EU(S)`` hold for every
coalition. :class:`ScalarGame` is the base for single-valued games such as the
conformal set size or an out-of-distribution score.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .adapters import as_member_predictor
from .imputation import Sampler, get_sampler
from .measures import NOTIONS, Decomposition, Measure, get_measure, to_moments

GAMES = (*NOTIONS, "prediction")
AGGREGATIONS = ("member_mean", "pointwise", "pooled")
MODES = ("uncertainty", "information_gain")


class ImputationGame:
    """Shared machinery: feature removal by imputation, batching, cache per coalition.

    Subclasses implement :meth:`_chunk_values`.

    Args:
        x: The point to explain, shape ``(d,)``.
        background: Background data used for imputation, shape ``(n, d)``.
        imputer: ``"marginal"``, ``"gaussian"``, ``"knn"`` or a ``Sampler``.
        n_samples: Number of imputations K per coalition.
        random_state: Seed for the imputation draws.
        batch_size: Maximum number of rows per model call.
    """

    def __init__(
        self,
        x: np.ndarray,
        background: np.ndarray,
        *,
        imputer: str | Sampler = "marginal",
        n_samples: int = 100,
        random_state: int | None = 0,
        batch_size: int = 50_000,
    ) -> None:
        self.x = np.asarray(x, dtype=float).ravel()
        self.sampler = get_sampler(imputer, background, n_samples, random_state)
        if self.x.shape[0] != self.sampler.n_features:
            msg = f"x has {self.x.shape[0]} features, background has {self.sampler.n_features}."
            raise ValueError(msg)
        self.n_players = self.x.shape[0]
        self.batch_size = batch_size
        self._cache: dict[bytes, np.ndarray] = {}

    def _chunk_values(self, points: list[np.ndarray]) -> list[np.ndarray]:
        """Raw value vector for each array of imputed points ``(K, d)``."""
        raise NotImplementedError

    def _raw(self, coalitions: np.ndarray) -> np.ndarray:
        """Raw values of shape ``(n_coalitions, n_values)``, cached per coalition."""
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
            for i, values in zip(chunk, self._chunk_values(points), strict=True):
                self._cache[keys[i]] = np.asarray(values, dtype=float)
        return np.stack([self._cache[k] for k in keys])

    @property
    def n_evaluated(self) -> int:
        """Number of distinct coalitions evaluated so far."""
        return len(self._cache)


class UncertaintyGame(ImputationGame):
    """Cooperative game over features whose value is a model's uncertainty.

    Args:
        model: Member predictor, sklearn ensemble or list of models (see
            :func:`shapiq_probly.adapters.as_member_predictor`).
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
        measure: ``"auto"`` (entropy for classification, variance for
            regression; for a bare callable the task is guessed from its
            output, so pass the measure explicitly if in doubt),
            ``"entropy"``, ``"credal"``, ``"variance"``, a
            :class:`~shapiq_probly.measures.Measure` or a classification
            callable ``(M, ..., C) -> (TU, AU, EU)``.
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
        measure: str | Measure | Decomposition = "auto",
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
        super().__init__(
            x, background, imputer=imputer, n_samples=n_samples, random_state=random_state, batch_size=batch_size
        )
        self.predict_members = as_member_predictor(model)
        self.aggregation = aggregation
        self.mode = mode

        probe = np.asarray(self.predict_members(self.x[None, :]), dtype=float)
        hint = getattr(self.predict_members, "task", None)
        if isinstance(measure, str) and measure == "auto":
            task = hint or self._guess_task(probe)
            measure = "variance" if task == "regression" else "entropy"
        self.measure = get_measure(measure)
        self.task = self.measure.task
        if hint is not None and hint != self.task:
            msg = f"Measure {self.measure.name!r} is for {self.task}, but the model is a {hint} model."
            raise ValueError(msg)

        # Regression targets are centred at the prediction for x: variances do not
        # change, but second moments stay small and numerically harmless.
        self._offset = 0.0
        if self.task == "regression":
            self._offset = float(self._represent(probe, 1)[..., 0].mean())
        rep = self._represent(probe, 1)[:, 0, :]
        if self.task == "classification" and not self._looks_like_probabilities(rep[:, None, :]):
            msg = (
                "The member predictor does not return class probabilities (non-negative, summing to 1). "
                "For regression pass measure='variance'."
            )
            raise ValueError(msg)
        self.n_members = rep.shape[0]
        #: Mean predictive distribution at ``x`` and its top class (classification only).
        self.predicted_proba: np.ndarray | None = None
        self.predicted_class: int | None = None
        if self.task == "classification":
            self.predicted_proba = rep.mean(axis=0)
            self.predicted_class = int(self.predicted_proba.argmax())
            #: What the prediction game explains: P(predicted class) or the mean prediction.
            self.prediction_value = float(self.predicted_proba[self.predicted_class])
        else:
            self.prediction_value = float(rep[:, 0].mean() + self._offset)
        self._empty = self._raw(np.zeros((1, self.n_players), dtype=bool))[0]

    @staticmethod
    def _looks_like_probabilities(out: np.ndarray) -> bool:
        return out.ndim == 3 and bool((out >= 0).all()) and bool(np.allclose(out.sum(axis=-1), 1.0, atol=1e-6))

    @classmethod
    def _guess_task(cls, probe: np.ndarray) -> str:
        """Task of an untagged callable, judged from its output for ``x``."""
        if probe.ndim == 2:
            return "regression"
        if cls._looks_like_probabilities(probe) or probe.ndim != 3 or probe.shape[-1] != 2:
            return "classification"
        return "regression"  # (M, n, 2) that is not a probability vector: mean and variance

    def _represent(self, out: np.ndarray, n_rows: int) -> np.ndarray:
        """Member predictions in a form where mixing is averaging, shape ``(M, n, D)``."""
        out = np.asarray(out, dtype=float)
        bad_shape = (
            "The member predictor must return shape (M, n, C) for classification or (M, n) / (M, n, 2) "
            f"for regression with n = {n_rows} rows, got {out.shape}."
        )
        if out.ndim < 2 or out.shape[1] != n_rows:
            raise ValueError(bad_shape)
        if self.task == "classification":
            if out.ndim != 3:
                raise ValueError(bad_shape)
            return out
        if out.ndim == 2:
            mean, variance = out, 0.0
        elif out.ndim == 3 and out.shape[-1] == 2:
            mean, variance = out[..., 0], out[..., 1]
        elif out.ndim == 3 and out.shape[-1] == 1:
            mean, variance = out[..., 0], 0.0
        else:
            raise ValueError(bad_shape)
        return to_moments(mean - self._offset, variance)

    def _raw_batch(self, rep: np.ndarray) -> np.ndarray:
        """Raw values (TU, AU, EU, prediction) for a batch of coalitions.

        Args:
            rep: Member predictions of shape ``(M, B, K, D)`` for ``B`` coalitions
                with ``K`` imputed points each.

        Returns:
            Array of shape ``(B, 4)``.
        """
        n_members, n_coalitions, n_points, dim = rep.shape
        decompose = self.measure.decompose
        if self.aggregation == "member_mean":
            parts = decompose(rep.mean(axis=2))
        elif self.aggregation == "pointwise":
            flat = decompose(rep.reshape(n_members, n_coalitions * n_points, dim))
            parts = [np.reshape(v, (n_coalitions, n_points)).mean(axis=1) for v in flat]
        else:  # pooled: members x imputations form one large ensemble per coalition
            parts = decompose(rep.transpose(0, 2, 1, 3).reshape(n_members * n_points, n_coalitions, dim))
        if self.task == "classification":
            prediction = rep[..., self.predicted_class].mean(axis=(0, 2))
        else:
            prediction = rep[..., 0].mean(axis=(0, 2)) + self._offset
        return np.column_stack([*(np.reshape(v, n_coalitions) for v in parts), prediction])

    def _chunk_values(self, points: list[np.ndarray]) -> list[np.ndarray]:
        sizes = np.array([len(p) for p in points])
        rep = self._represent(self.predict_members(np.concatenate(points)), int(sizes.sum()))
        if (sizes == sizes[0]).all():
            return list(self._raw_batch(rep.reshape(rep.shape[0], len(points), int(sizes[0]), rep.shape[-1])))
        # A custom sampler may return a different number of points per coalition.
        return [self._raw_batch(r[:, None])[0] for r in np.split(rep, np.cumsum(sizes)[:-1], axis=1)]

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


class ScalarGame(ImputationGame):
    """Game with a single value per coalition, computed from the imputed points.

    ``v(S) = value(points)`` where ``points`` of shape ``(K, d)`` agree with
    ``x`` on the coalition and are imputed elsewhere. Subclasses may instead
    override :meth:`_chunk_values` to batch the work over coalitions.

    Args:
        x: The point to explain, shape ``(d,)``.
        background: Background data used for imputation, shape ``(n, d)``.
        value: Callable mapping imputed points ``(K, d)`` to a float.
        name: Name of the explained quantity, for display.
        imputer: ``"marginal"``, ``"gaussian"``, ``"knn"`` or a ``Sampler``.
        n_samples: Number of imputations K per coalition.
        random_state: Seed for the imputation draws.
        batch_size: Maximum number of rows per model call.
    """

    def __init__(
        self,
        x: np.ndarray,
        background: np.ndarray,
        value: Callable[[np.ndarray], float] | None = None,
        *,
        name: str = "value",
        imputer: str | Sampler = "marginal",
        n_samples: int = 100,
        random_state: int | None = 0,
        batch_size: int = 50_000,
    ) -> None:
        super().__init__(
            x, background, imputer=imputer, n_samples=n_samples, random_state=random_state, batch_size=batch_size
        )
        self.name = name
        self._value = value
        #: Extra facts about the game that end up in the explanation.
        self.info: dict[str, Any] = {}

    def _chunk_values(self, points: list[np.ndarray]) -> list[np.ndarray]:
        if self._value is None:
            msg = "Pass 'value' or override '_chunk_values'."
            raise NotImplementedError(msg)
        return [np.array([float(self._value(p))]) for p in points]

    def value_function(self, coalitions: np.ndarray, notion: str | None = None) -> np.ndarray:
        """Game values for a boolean coalition matrix ``(n, d)``; ``notion`` is ignored."""
        coalitions = np.atleast_2d(np.asarray(coalitions, dtype=bool))
        return self._raw(coalitions)[:, 0]

    def __call__(self, coalitions: np.ndarray, notion: str | None = None) -> np.ndarray:
        return self.value_function(coalitions, notion)
