"""Integration with shapiq. Skipped unless shapiq is installed."""

import numpy as np
import pytest

shapiq = pytest.importorskip("shapiq")

from shapiq_probly import (  # noqa: E402
    OODGame,
    UncertaintyGame,
    all_coalitions,
    compute_index,
    explain_uncertainty,
)
from shapiq_probly.shapiq_bridge import as_shapiq_game, explanation_to_shapiq  # noqa: E402

X0 = np.array([0.8, 0.8, 1.2, 0.5])


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_exact_computer_agrees_with_our_oracle(toy_model, toy_background, notion):
    game = UncertaintyGame(toy_model, X0, toy_background, n_samples=30)
    ours = compute_index(game(all_coalitions(4), notion), "k-SII", 2)
    computer = shapiq.ExactComputer(game=as_shapiq_game(game, notion), n_players=4)
    theirs = computer(index="k-SII", order=2)
    for interaction, value in ours.items():
        assert theirs[interaction] == pytest.approx(value, abs=1e-8)


def test_additivity_with_shapiq_arithmetic(toy_model, toy_background):
    expl = explain_uncertainty(toy_model, X0, toy_background, n_samples=30)
    iv = explanation_to_shapiq(expl)
    diff = iv["total"] - (iv["aleatoric"] + iv["epistemic"])
    assert np.abs(diff.values).max() < 1e-8


def test_scalar_games_can_be_wrapped_too(toy_background):
    game = OODGame(X0, toy_background, n_samples=20)
    ours = compute_index(game.value_function(all_coalitions(4)), "SV")
    theirs = shapiq.ExactComputer(game=as_shapiq_game(game), n_players=4)(index="SV", order=1)
    for interaction, value in ours.items():
        assert theirs[interaction] == pytest.approx(value, abs=1e-8)
