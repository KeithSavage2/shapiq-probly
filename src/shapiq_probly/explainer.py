"""Single entry point: explain a prediction and its uncertainty."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .exact import all_coalitions, compute_index, permutation_shapley
from .explanation import Attribution, UncertaintyExplanation
from .games import GAMES, UncertaintyGame
from .imputation import Sampler
from .measures import NOTIONS, Decomposition


class UncertaintyExplainer:
    """Explains predictions and their total, aleatoric and epistemic uncertainty.

    Up to ``max_exact_features`` features all ``2^n`` coalitions are evaluated
    once and every index is exact. Beyond that, Shapley values are approximated
    by permutation sampling (interactions for many features: see
    ``shapiq_probly.shapiq_bridge``).

    Args:
        model: Member predictor ``X -> (M, n, C)``, sklearn ensemble or list of
            classifiers.
        background: Background data for imputation, shape ``(n, d)``.
        index: ``"k-SII"`` (default), ``"SV"``, ``"SII"`` or ``"Moebius"``.
        max_order: Largest interaction size.
        imputer: ``"marginal"``, ``"gaussian"``, ``"knn"`` or a ``Sampler``.
        aggregation: ``"member_mean"`` (A), ``"pointwise"`` (B) or ``"pooled"`` (C).
        mode: ``"uncertainty"`` or ``"information_gain"``.
        measure: Uncertainty decomposition, ``"entropy"`` or a callable.
        n_samples: Imputations per coalition.
        feature_names: Optional feature names.
        budget: Coalition evaluations for the approximation (default ``200 * d``).
        max_exact_features: Largest ``d`` for exact computation.
        random_state: Seed.
    """

    def __init__(
        self,
        model: Any,
        background: np.ndarray,
        *,
        index: str = "k-SII",
        max_order: int = 2,
        imputer: str | Sampler = "marginal",
        aggregation: str = "member_mean",
        mode: str = "uncertainty",
        measure: str | Decomposition = "entropy",
        n_samples: int = 100,
        feature_names: list[str] | None = None,
        budget: int | None = None,
        max_exact_features: int = 12,
        random_state: int | None = 0,
    ) -> None:
        self.model = model
        self.background = np.asarray(background, dtype=float)
        self.index = index
        self.max_order = 1 if index == "SV" else max_order
        self.imputer = imputer
        self.aggregation = aggregation
        self.mode = mode
        self.measure = measure
        self.n_samples = n_samples
        d = self.background.shape[1]
        self.feature_names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(d)]
        if len(self.feature_names) != d:
            msg = f"Got {len(self.feature_names)} feature names for {d} features."
            raise ValueError(msg)
        self.budget = budget
        self.max_exact_features = max_exact_features
        self.random_state = random_state

    def game(self, x: np.ndarray) -> UncertaintyGame:
        """The underlying game for ``x``."""
        return UncertaintyGame(
            self.model,
            x,
            self.background,
            imputer=self.imputer,
            aggregation=self.aggregation,
            mode=self.mode,
            measure=self.measure,
            n_samples=self.n_samples,
            random_state=self.random_state,
        )

    def explain(self, x: np.ndarray) -> UncertaintyExplanation:
        """Explain a single point ``x`` of shape ``(d,)``."""
        game = self.game(x)
        d = game.n_players
        exact = d <= self.max_exact_features
        attributions: dict[str, Attribution] = {}
        shapley: dict[str, np.ndarray] = {}
        game_values = None
        index, max_order = self.index, min(self.max_order, d)

        if exact:
            game_values = game.evaluate(all_coalitions(d))
            for name in GAMES:
                vals = game_values[name]
                shapley[name] = np.array(list(compute_index(vals, "SV").values()))
                attributions[name] = Attribution(
                    name, index, max_order, compute_index(vals, index, max_order), float(vals[0]), float(vals[-1])
                )
        else:
            budget = self.budget or 200 * d
            warnings.warn(
                f"{d} features > max_exact_features={self.max_exact_features}: approximating Shapley values "
                f"with about {budget} coalition evaluations ({budget * game.sampler.n_samples} model rows per "
                "member). Interactions are not computed.",
                stacklevel=2,
            )

            def joint(coalitions: np.ndarray) -> np.ndarray:
                ev = game.evaluate(coalitions)
                return np.column_stack([ev[name] for name in GAMES])

            phi = permutation_shapley(joint, d, budget, self.random_state)
            ends = joint(np.array([np.zeros(d, dtype=bool), np.ones(d, dtype=bool)]))
            for j, name in enumerate(GAMES):
                shapley[name] = phi[:, j]
                values = {(i,): float(phi[i, j]) for i in range(d)}
                attributions[name] = Attribution(name, "SV", 1, values, float(ends[0, j]), float(ends[1, j]), True)

        ends_raw = game._raw(np.array([np.zeros(d, dtype=bool), np.ones(d, dtype=bool)]))
        return UncertaintyExplanation(
            x=game.x,
            feature_names=self.feature_names,
            predicted_proba=game.predicted_proba,
            predicted_class=game.predicted_class,
            uncertainty={n: float(ends_raw[1, j]) for j, n in enumerate(NOTIONS)},
            uncertainty_empty={n: float(ends_raw[0, j]) for j, n in enumerate(NOTIONS)},
            attributions=attributions,
            shapley=shapley,
            game_values=game_values,
            config={
                "index": attributions["total"].index,
                "max_order": attributions["total"].max_order,
                "imputer": self.imputer if isinstance(self.imputer, str) else type(self.imputer).__name__,
                "aggregation": self.aggregation,
                "mode": self.mode,
                "measure": self.measure
                if isinstance(self.measure, str)
                else getattr(self.measure, "__name__", "custom"),
                "n_samples": game.sampler.n_samples,
                "n_members": game.n_members,
                "exact": exact,
            },
        )

    def explain_many(self, X: np.ndarray) -> list[UncertaintyExplanation]:
        """Explain every row of ``X``."""
        return [self.explain(x) for x in np.atleast_2d(X)]


def explain_uncertainty(model: Any, x: np.ndarray, background: np.ndarray, **kwargs: Any) -> UncertaintyExplanation:
    """Explain the prediction for ``x`` and its uncertainty in one call.

    Keyword arguments are forwarded to :class:`UncertaintyExplainer`.

    Example:
        >>> expl = explain_uncertainty(forest, x, background=X_train)  # doctest: +SKIP
        >>> print(expl.summary())  # doctest: +SKIP
    """
    return UncertaintyExplainer(model, background, **kwargs).explain(x)
