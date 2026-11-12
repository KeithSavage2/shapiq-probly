"""Additive uncertainty decompositions on second-order (member-wise) predictions.

Every decomposition takes an array whose *first* axis indexes the ensemble
members (or dropout samples) and whose *last* axis holds a representation of
one predictive distribution in which **mixing distributions is averaging**:

* classification: class probabilities, shape ``(M, ..., C)``;
* regression: raw moments ``(E[y], E[y^2])``, shape ``(M, ..., 2)``.

That property is what lets the game "average distributions first, measure
afterwards" with one mechanism for both tasks. Each decomposition returns total,
aleatoric and epistemic uncertainty of shape ``(...)`` with
``total == aleatoric + epistemic``.

The entropy and variance decompositions mirror ``SecondOrderEntropyDecomposition``
and ``SecondOrderVarianceDecomposition`` in ``probly.quantification``; the credal
one follows ``CredalSetEntropyDecomposition`` (upper minus lower entropy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Decomposition = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]

NOTIONS = ("total", "aleatoric", "epistemic")
_TINY = 1e-300


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats, with the convention ``0 * log(0) = 0``."""
    p = np.asarray(probs, dtype=float)
    logp = np.log(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=axis) + 0.0  # + 0.0 turns -0.0 into 0.0


def entropy_decomposition(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose predictive entropy into aleatoric and epistemic parts.

    Total = entropy of the mean distribution, aleatoric = mean member entropy,
    epistemic = mutual information.

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


def to_moments(mean: np.ndarray, variance: np.ndarray | float = 0.0) -> np.ndarray:
    """Stack mean and variance into raw moments ``(E[y], E[y^2])`` on a new last axis."""
    mean = np.asarray(mean, dtype=float)
    return np.stack([mean, np.asarray(variance, dtype=float) + mean**2], axis=-1)


def variance_decomposition(moments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose predictive variance (law of total variance).

    Total = variance of the mixture over members, aleatoric = mean member
    variance, epistemic = variance of the member means.

    Args:
        moments: Raw moments ``(E[y], E[y^2])`` per member, shape ``(M, ..., 2)``.

    Returns:
        ``(total, aleatoric, epistemic)``, each of shape ``(...)``.
    """
    m = np.asarray(moments, dtype=float)
    first, second = m[..., 0], m[..., 1]
    total = second.mean(axis=0) - first.mean(axis=0) ** 2
    aleatoric = (second - first**2).mean(axis=0)
    epistemic = total - aleatoric
    return total, aleatoric, epistemic


def _line_search(q: np.ndarray, direction: np.ndarray, gamma_max: np.ndarray, n_iter: int = 40) -> np.ndarray:
    """Maximise the entropy of ``q + gamma * direction`` over ``[0, gamma_max]`` by bisection."""
    lo, hi = np.zeros_like(gamma_max), gamma_max.copy()
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        point = np.maximum(q + mid[:, None] * direction, _TINY)
        slope = -(direction * np.log(point)).sum(axis=1)
        lo = np.where(slope > 0, mid, lo)
        hi = np.where(slope > 0, hi, mid)
    return (lo + hi) / 2


def upper_entropy(probs: np.ndarray, max_iter: int = 2000, tol: float = 1e-9) -> np.ndarray:
    """Largest entropy within the convex hull of the member distributions.

    For two classes there is a closed form. Otherwise the concave problem
    ``max_w H(sum_m w_m p_m)`` over the simplex is solved with away-step
    Frank-Wolfe until the duality gap is below ``tol``.

    Args:
        probs: Class probabilities of shape ``(M, ..., C)``.
        max_iter: Iteration cap for the multi-class solver.
        tol: Duality-gap tolerance in nats.

    Returns:
        Upper entropy of shape ``(...)``.
    """
    p = np.asarray(probs, dtype=float)
    n_members, n_classes = p.shape[0], p.shape[-1]
    if n_classes == 2:
        closest = np.clip(0.5, p[..., 1].min(axis=0), p[..., 1].max(axis=0))
        return entropy(np.stack([1 - closest, closest], axis=-1))

    batch_shape = p.shape[1:-1]
    members = np.moveaxis(p.reshape(n_members, -1, n_classes), 0, 1)  # (B, M, C)
    weights = np.full(members.shape[:2], 1.0 / n_members)
    active = np.arange(len(members))  # problems that have not converged yet
    for _ in range(max_iter):
        if active.size == 0:
            break
        pts, w = members[active], weights[active]
        rows = np.arange(len(active))
        q = np.einsum("bm,bmc->bc", w, pts)
        # Gradient of the entropy with respect to the weights (up to a constant).
        grad = -np.einsum("bmc,bc->bm", pts, np.log(np.maximum(q, _TINY)))
        centre = (w * grad).sum(axis=1)
        toward = grad.argmax(axis=1)
        away = np.where(w > 0, grad, np.inf).argmin(axis=1)
        gap_toward = grad[rows, toward] - centre
        gap_away = centre - grad[rows, away]
        open_ = gap_toward > tol
        if not open_.any():
            break
        weight_away = w[rows, away]
        use_away = (gap_away > gap_toward) & (weight_away < 1)
        direction = np.where(use_away[:, None], q - pts[rows, away], pts[rows, toward] - q)
        gamma_max = np.where(use_away, weight_away / np.maximum(1 - weight_away, _TINY), 1.0)
        gamma = np.where(open_, _line_search(q, direction, gamma_max), 0.0)
        w = w * np.where(use_away, 1 + gamma, 1 - gamma)[:, None]
        w[rows, np.where(use_away, away, toward)] += np.where(use_away, -gamma, gamma)
        w = np.where(w < 1e-14, 0.0, w)
        weights[active] = w / w.sum(axis=1, keepdims=True)
        active = active[open_]
    q = np.einsum("bm,bmc->bc", weights, members)
    return entropy(q).reshape(batch_shape)


def credal_decomposition(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose uncertainty of the credal set spanned by the members.

    The credal set is the convex hull of the member distributions. Total =
    upper entropy, aleatoric = lower entropy, epistemic = their difference
    (the "width" of the credal set). Entropy is concave, so the lower entropy
    is attained at a member.

    Args:
        probs: Class probabilities of shape ``(M, ..., C)``.

    Returns:
        ``(total, aleatoric, epistemic)``, each of shape ``(...)``.
    """
    p = np.asarray(probs, dtype=float)
    total = upper_entropy(p)
    aleatoric = entropy(p).min(axis=0)
    epistemic = total - aleatoric
    return total, aleatoric, epistemic


@dataclass(frozen=True)
class Measure:
    """An uncertainty decomposition together with the task it applies to.

    Attributes:
        name: Identifier, e.g. ``"entropy"``.
        task: ``"classification"`` or ``"regression"``.
        decompose: The decomposition function.
        unit: Unit of the uncertainty values, for display.
    """

    name: str
    task: str
    decompose: Decomposition
    unit: str = ""


MEASURES: dict[str, Measure] = {
    "entropy": Measure("entropy", "classification", entropy_decomposition, "nats"),
    "credal": Measure("credal", "classification", credal_decomposition, "nats"),
    "variance": Measure("variance", "regression", variance_decomposition, "variance"),
}


def get_measure(measure: str | Measure | Decomposition) -> Measure:
    """Resolve a measure given by name, as a :class:`Measure` or as a classification callable."""
    if isinstance(measure, Measure):
        return measure
    if callable(measure):
        return Measure(getattr(measure, "__name__", "custom"), "classification", measure)
    try:
        return MEASURES[measure]
    except KeyError:
        msg = f"Unknown measure {measure!r}. Available: {sorted(MEASURES)} or 'auto'."
        raise ValueError(msg) from None
