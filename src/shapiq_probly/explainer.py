"""Entry points: explain a prediction and its uncertainty, or any scalar game."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .conformal import ConformalSetSizeGame
from .exact import all_coalitions, compute_index, permutation_shapley
from .explanation import Attribution, ScalarExplanation, UncertaintyExplanation
from .games import GAMES, ScalarGame, UncertaintyGame
from .imputation import Sampler
from .measures import NOTIONS, Decomposition, Measure
from .ood import OODGame, Score


def _feature_names(feature_names: list[str] | None, d: int) -> list[str]:
    names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(d)]
    if len(names) != d:
        msg = f"Got {len(names)} feature names for {d} features."
        raise ValueError(msg)
    return names


def _ends(d: int) -> np.ndarray:
    return np.array([np.zeros(d, dtype=bool), np.ones(d, dtype=bool)])


def _warn_approximation(d: int, max_exact: int, budget: int, n_samples: int) -> None:
    warnings.warn(
        f"{d} features > max_exact_features={max_exact}: approximating Shapley values with about {budget} "
        f"coalition evaluations ({budget * n_samples} imputed rows). Interactions are not computed.",
        stacklevel=3,
    )


class UncertaintyExplainer:
    """Explains predictions and their total, aleatoric and epistemic uncertainty.

    Works for classification (entropy or credal decomposition) and regression
    (variance decomposition). Up to ``max_exact_features`` features all ``2^n``
    coalitions are evaluated once and every index is exact. Beyond that, Shapley
    values are approximated by permutation sampling (interactions for many
    features: see ``shapiq_probly.shapiq_bridge``).

    Args:
        model: Member predictor, sklearn ensemble or list of models.
        background: Background data for imputation, shape ``(n, d)``.
        index: ``"k-SII"`` (default), ``"SV"``, ``"SII"`` or ``"Moebius"``.
        max_order: Largest interaction size.
        imputer: ``"marginal"``, ``"gaussian"``, ``"knn"`` or a ``Sampler``.
        aggregation: ``"member_mean"`` (A), ``"pointwise"`` (B) or ``"pooled"`` (C).
        mode: ``"uncertainty"`` or ``"information_gain"``.
        measure: ``"auto"``, ``"entropy"``, ``"credal"``, ``"variance"``, a
            ``Measure`` or a classification callable.
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
        measure: str | Measure | Decomposition = "auto",
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
        self.feature_names = _feature_names(feature_names, self.background.shape[1])
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
            _warn_approximation(d, self.max_exact_features, budget, game.sampler.n_samples)

            def joint(coalitions: np.ndarray) -> np.ndarray:
                ev = game.evaluate(coalitions)
                return np.column_stack([ev[name] for name in GAMES])

            phi = permutation_shapley(joint, d, budget, self.random_state)
            ends = joint(_ends(d))
            for j, name in enumerate(GAMES):
                shapley[name] = phi[:, j]
                values = {(i,): float(phi[i, j]) for i in range(d)}
                attributions[name] = Attribution(name, "SV", 1, values, float(ends[0, j]), float(ends[1, j]), True)

        ends_raw = game._raw(_ends(d))
        return UncertaintyExplanation(
            x=game.x,
            feature_names=self.feature_names,
            predicted_proba=game.predicted_proba,
            predicted_class=game.predicted_class,
            task=game.task,
            prediction_value=game.prediction_value,
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
                "measure": game.measure.name,
                "unit": game.measure.unit,
                "n_samples": game.sampler.n_samples,
                "n_members": game.n_members,
                "exact": exact,
            },
        )

    def explain_many(self, X: np.ndarray) -> list[UncertaintyExplanation]:
        """Explain every row of ``X`` (input for the global plots)."""
        return [self.explain(x) for x in np.atleast_2d(X)]


def explain_uncertainty(model: Any, x: np.ndarray, background: np.ndarray, **kwargs: Any) -> UncertaintyExplanation:
    """Explain the prediction for ``x`` and its uncertainty in one call.

    Keyword arguments are forwarded to :class:`UncertaintyExplainer`.

    Example:
        >>> expl = explain_uncertainty(forest, x, background=X_train)  # doctest: +SKIP
        >>> print(expl.summary())  # doctest: +SKIP
    """
    return UncertaintyExplainer(model, background, **kwargs).explain(x)


def explain_game(
    game: ScalarGame,
    *,
    index: str = "k-SII",
    max_order: int = 2,
    feature_names: list[str] | None = None,
    budget: int | None = None,
    max_exact_features: int = 12,
    random_state: int | None = 0,
) -> ScalarExplanation:
    """Explain a single-valued game (conformal set size, OOD score, custom).

    Args:
        game: The game to explain.
        index: ``"k-SII"`` (default), ``"SV"``, ``"SII"`` or ``"Moebius"``.
        max_order: Largest interaction size.
        feature_names: Optional feature names.
        budget: Coalition evaluations for the approximation (default ``200 * d``).
        max_exact_features: Largest ``d`` for exact computation.
        random_state: Seed of the approximation.
    """
    d = game.n_players
    names = _feature_names(feature_names, d)
    max_order = 1 if index == "SV" else min(max_order, d)
    if d <= max_exact_features:
        values = game.value_function(all_coalitions(d))
        shapley = np.array(list(compute_index(values, "SV").values()))
        attribution = Attribution(
            game.name, index, max_order, compute_index(values, index, max_order), float(values[0]), float(values[-1])
        )
    else:
        budget = budget or 200 * d
        _warn_approximation(d, max_exact_features, budget, game.sampler.n_samples)
        values = None
        shapley = permutation_shapley(lambda c: game.value_function(c)[:, None], d, budget, random_state)[:, 0]
        ends = game.value_function(_ends(d))
        approx = {(i,): float(shapley[i]) for i in range(d)}
        attribution = Attribution(game.name, "SV", 1, approx, float(ends[0]), float(ends[1]), True)
    return ScalarExplanation(
        name=game.name,
        x=game.x,
        feature_names=names,
        value=attribution.full_value,
        baseline=attribution.baseline_value,
        attribution=attribution,
        shapley=shapley,
        game_values=values,
        info=dict(game.info),
    )


def explain_set_size(
    model: Any,
    x: np.ndarray,
    background: np.ndarray,
    calibration: tuple[np.ndarray, np.ndarray],
    *,
    alpha: float = 0.1,
    aggregation: str = "pointwise",
    imputer: str | Sampler = "marginal",
    n_samples: int = 100,
    random_state: int | None = 0,
    **kwargs: Any,
) -> ScalarExplanation:
    """Explain the size of the conformal prediction set at ``x`` (classification).

    See :class:`~shapiq_probly.conformal.ConformalSetSizeGame` for the game
    arguments; further keyword arguments go to :func:`explain_game`.
    """
    game = ConformalSetSizeGame(
        model,
        x,
        background,
        calibration,
        alpha=alpha,
        aggregation=aggregation,
        imputer=imputer,
        n_samples=n_samples,
        random_state=random_state,
    )
    return explain_game(game, random_state=random_state, **kwargs)


def explain_ood(
    x: np.ndarray,
    background: np.ndarray,
    *,
    score: str | Score = "knn",
    reference: np.ndarray | None = None,
    k: int = 10,
    imputer: str | Sampler = "marginal",
    n_samples: int = 100,
    random_state: int | None = 0,
    **kwargs: Any,
) -> ScalarExplanation:
    """Explain an out-of-distribution score of ``x``.

    See :class:`~shapiq_probly.ood.OODGame` for the game arguments; further
    keyword arguments go to :func:`explain_game`.
    """
    game = OODGame(
        x,
        background,
        score=score,
        reference=reference,
        k=k,
        imputer=imputer,
        n_samples=n_samples,
        random_state=random_state,
    )
    return explain_game(game, random_state=random_state, **kwargs)
