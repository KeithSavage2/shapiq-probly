"""Bridge to shapiq (optional dependency, ``pip install shapiq-probly[shapiq]``).

Checked in the CI job ``shapiq`` against shapiq 1.8.0:
``tests/test_shapiq_bridge.py`` compares with ``shapiq.ExactComputer`` and is
skipped when shapiq is missing. Approximators and shapiq plots are not covered yet.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .explanation import Attribution, UncertaintyExplanation
from .games import ScalarGame, UncertaintyGame


def as_shapiq_game(
    game: UncertaintyGame | ScalarGame,
    notion: str = "epistemic",
    *,
    normalize: bool = False,
    player_names: list[str] | None = None,
) -> Any:
    """Wrap one notion of an :class:`UncertaintyGame` (or a scalar game) as a ``shapiq.Game``.

    The result can be passed to ``shapiq.ExactComputer``, any shapiq
    approximator or ``shapiq.AgnosticExplainer``. All notions wrapped from the
    same ``game`` share its coalition cache.

    Args:
        game: The uncertainty game, or a ``ScalarGame`` such as the conformal
            set size or an OOD score.
        notion: ``"total"``, ``"aleatoric"``, ``"epistemic"`` or ``"prediction"``
            (ignored for scalar games).
        normalize: Whether shapiq should subtract ``v(empty)``.
        player_names: Optional feature names.
    """
    import shapiq  # noqa: PLC0415

    empty = float(game.value_function(np.zeros((1, game.n_players), dtype=bool), notion)[0])

    class ShapiqUncertaintyGame(shapiq.Game):
        def __init__(self) -> None:
            super().__init__(
                n_players=game.n_players,
                normalize=normalize,
                normalization_value=empty,
                player_names=player_names,
            )

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            return game.value_function(coalitions, notion)

    return ShapiqUncertaintyGame()


def to_interaction_values(attribution: Attribution, n_players: int) -> Any:
    """Convert an :class:`Attribution` into ``shapiq.InteractionValues`` (for shapiq plots)."""
    import shapiq  # noqa: PLC0415

    lookup = {interaction: pos for pos, interaction in enumerate(attribution.values)}
    return shapiq.InteractionValues(
        values=np.array(list(attribution.values.values()), dtype=float),
        index=attribution.index,
        max_order=attribution.max_order,
        min_order=1,
        n_players=n_players,
        interaction_lookup=lookup,
        estimated=attribution.estimated,
        baseline_value=attribution.baseline_value,
    )


def explanation_to_shapiq(explanation: UncertaintyExplanation) -> dict[str, Any]:
    """All four attributions of an explanation as ``shapiq.InteractionValues``."""
    n = explanation.n_features
    return {name: to_interaction_values(a, n) for name, a in explanation.attributions.items()}
