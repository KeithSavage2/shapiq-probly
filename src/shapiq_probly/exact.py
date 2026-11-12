"""Exact game-theoretic indices from all 2^n coalition values.

Coalitions are indexed by bitmask: coalition ``S`` has index ``sum(2**i for i in S)``.
This module is our own small "truth oracle"; with shapiq installed the same
numbers can be obtained from ``shapiq.ExactComputer`` (see ``shapiq_bridge``).

All indices are linear in the game values, which is why
``phi(TU) = phi(AU) + phi(EU)`` follows from ``TU = AU + EU``.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np
from scipy.special import bernoulli

Interaction = tuple[int, ...]

INDICES = ("SV", "k-SII", "SII", "Moebius")


def all_coalitions(n_players: int) -> np.ndarray:
    """Boolean matrix ``(2^n, n)`` of all coalitions in bitmask order."""
    masks = np.arange(2**n_players)[:, None]
    return ((masks >> np.arange(n_players)) & 1).astype(bool)


def _mask(interaction: Interaction) -> int:
    return sum(1 << i for i in interaction)


def moebius_transform(values: np.ndarray) -> np.ndarray:
    """Moebius transform: ``m(S) = sum_{T subset S} (-1)^{|S|-|T|} v(T)``.

    It is the complete, exact decomposition of the game: ``v(S) = sum_{T subset S} m(T)``.
    """
    m = np.array(values, dtype=float)
    n = int(np.log2(len(m)))
    if 2**n != len(m):
        msg = "values must have length 2^n."
        raise ValueError(msg)
    m = m.reshape((2,) * n) if n else m
    for axis in range(n):
        upper = [slice(None)] * n
        lower = [slice(None)] * n
        upper[axis], lower[axis] = 1, 0
        m[tuple(upper)] -= m[tuple(lower)]
    # reshape((2,)*n) puts player n-1 on axis 0, so flattening restores bitmask order.
    return m.reshape(-1)


def _sizes(n: int) -> np.ndarray:
    masks = np.arange(2**n)
    return ((masks[:, None] >> np.arange(n)) & 1).sum(axis=1)


def shapley_interactions(values: np.ndarray, max_order: int) -> dict[Interaction, float]:
    """Shapley interaction index (SII) for all interactions of size 1..max_order.

    Uses ``SII(S) = sum_{T superset S} m(T) / (|T| - |S| + 1)``. For ``|S| = 1``
    this is the Shapley value.
    """
    m = moebius_transform(values)
    n = int(np.log2(len(m)))
    sizes = _sizes(n)
    masks = np.arange(2**n)
    out: dict[Interaction, float] = {}
    for order in range(1, min(max_order, n) + 1):
        weights = m / (sizes - order + 1.0).clip(min=1)
        for interaction in combinations(range(n), order):
            mask = _mask(interaction)
            out[interaction] = float(weights[(masks & mask) == mask].sum())
    return out


def k_shapley_interactions(values: np.ndarray, max_order: int) -> dict[Interaction, float]:
    """k-SII (n-Shapley values): SII aggregated so that all values sum to ``v(N) - v(empty)``.

    ``Phi_k(S) = sum_{T superset S, |T| <= k} B_{|T|-|S|} SII(T)`` with Bernoulli numbers B.
    """
    sii = shapley_interactions(values, max_order)
    bern = bernoulli(max_order)
    out: dict[Interaction, float] = {}
    for s in sii:
        total = 0.0
        for t, val in sii.items():
            if len(t) >= len(s) and set(s) <= set(t):
                total += bern[len(t) - len(s)] * val
        out[s] = total
    return out


def compute_index(values: np.ndarray, index: str = "k-SII", max_order: int = 2) -> dict[Interaction, float]:
    """Compute an interaction index from all 2^n game values (bitmask order).

    Args:
        values: Game values of shape ``(2^n,)``.
        index: ``"SV"``, ``"k-SII"``, ``"SII"`` or ``"Moebius"``.
        max_order: Largest interaction size (ignored for ``"SV"``).

    Returns:
        Mapping from interaction (sorted tuple of player indices) to value.
    """
    values = np.asarray(values, dtype=float)
    n = int(np.log2(len(values)))
    if index == "SV":
        return shapley_interactions(values, 1)
    if index == "SII":
        return shapley_interactions(values, max_order)
    if index == "k-SII":
        return k_shapley_interactions(values, max_order)
    if index == "Moebius":
        m = moebius_transform(values)
        return {
            s: float(m[_mask(s)]) for order in range(1, min(max_order, n) + 1) for s in combinations(range(n), order)
        }
    msg = f"Unknown index {index!r}. Available: {INDICES}."
    raise ValueError(msg)


def permutation_shapley(
    game_values: Callable[[np.ndarray], np.ndarray],
    n_players: int,
    budget: int,
    random_state: int | None = 0,
) -> np.ndarray:
    """Approximate Shapley values by permutation sampling (with antithetic pairs).

    Args:
        game_values: Callable mapping a coalition matrix ``(b, n)`` to values
            ``(b, g)`` for ``g`` games evaluated jointly.
        n_players: Number of players.
        budget: Approximate number of coalition evaluations.
        random_state: Seed.

    Returns:
        Array ``(n_players, g)`` of estimated Shapley values. Because all games
        share the permutations, additivity between games is preserved exactly.
    """
    rng = np.random.default_rng(random_state)
    n_perms = max(2, budget // n_players)
    n_perms += n_perms % 2
    phi = None
    for _ in range(n_perms // 2):
        perm = rng.permutation(n_players)
        for order in (perm, perm[::-1]):
            coalitions = np.zeros((n_players + 1, n_players), dtype=bool)
            for step, player in enumerate(order):
                coalitions[step + 1 :, player] = True
            vals = np.asarray(game_values(coalitions))
            if phi is None:
                phi = np.zeros((n_players, vals.shape[1]))
            phi[order] += np.diff(vals, axis=0)
    return phi / n_perms
