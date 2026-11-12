"""Additive uncertainty decompositions on second-order (member-wise) predictions.

All functions take an array of class probabilities whose *first* axis indexes
the ensemble members (or dropout samples) and whose *last* axis indexes classes,
i.e. shape ``(M, ..., C)``. They return total, aleatoric and epistemic
uncertainty with shape ``(...)`` such that ``total == aleatoric + epistemic``.

The entropy decomposition mirrors ``SecondOrderEntropyDecomposition`` in
``probly.quantification``: total = entropy of the mean distribution,
aleatoric = mean member entropy, epistemic = mutual information.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Decomposition = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]

NOTIONS = ("total", "aleatoric", "epistemic")


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats, with the convention ``0 * log(0) = 0``."""
    p = np.asarray(probs, dtype=float)
    logp = np.log(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=axis)


def entropy_decomposition(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose predictive entropy into aleatoric and epistemic parts.

    Args:
        probs: Class probabilities of shape ``(M, ..., C)``.

    Returns:
        ``(total, aleatoric, epistemic)``, each of shape ``(...)``.
    """
    p = np.asarray(probs, dtype=float)
    total = entropy(p.mean(axis=0))
    aleatoric = entropy(p).mean(axis=0)
    # Defined as the difference so that additivity holds to machine precision.
    epistemic = total - aleatoric
    return total, aleatoric, epistemic


DECOMPOSITIONS: dict[str, Decomposition] = {"entropy": entropy_decomposition}


def get_decomposition(measure: str | Decomposition) -> Decomposition:
    """Resolve a decomposition given by name or as a callable."""
    if callable(measure):
        return measure
    try:
        return DECOMPOSITIONS[measure]
    except KeyError:
        msg = f"Unknown measure {measure!r}. Available: {sorted(DECOMPOSITIONS)}."
        raise ValueError(msg) from None
