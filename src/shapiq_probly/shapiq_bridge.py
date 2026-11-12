"""Bridge to shapiq (optional dependency, ``pip install shapiq-probly[shapiq]``).

NOTE: this module was written against the documented shapiq 1.8 API but could
not be executed in the environment it was developed in. ``tests/test_shapiq_bridge.py``
checks it against ``shapiq.ExactComputer`` and is skipped when shapiq is missing.
Run it once in your own environment before relying on this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .explanation import Attribution, UncertaintyExplanation
from .games import UncertaintyGame


def as_shapiq_game(
    game: UncertaintyGame,
    notion: str = "epistemic",
    *,
    normalize: bool = False,
    player_names: list[str] | None = None,
) -> Any:
    """Wrap one notion of an :class:`UncertaintyGame` as a ``shapiq.Game``.

    The result can be passed to ``shapiq.ExactComputer``, any shapiq
    approximator or ``shapiq.AgnosticExplainer``. All notions wrapped from the
    same ``game`` share its coalition cache.

    Args:
        game: The uncertainty game.
        notion: ``"total"``, ``"aleatoric"``, ``"epistemic"`` or ``"prediction"``.
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
