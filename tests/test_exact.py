from itertools import combinations
from math import factorial

import numpy as np
import pytest

from shapiq_probly.exact import all_coalitions, compute_index, moebius_transform, permutation_shapley


def brute_force_shapley(values: np.ndarray, n: int) -> np.ndarray:
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            w = factorial(size) * factorial(n - size - 1) / factorial(n)
            for s in combinations(others, size):
                mask = sum(1 << j for j in s)
                phi[i] += w * (values[mask | (1 << i)] - values[mask])
    return phi


@pytest.fixture
def random_game():
    return np.random.default_rng(0).normal(size=2**5)


def test_all_coalitions_bitmask_order():
    c = all_coalitions(3)
    assert c.shape == (8, 3)
    assert c[5].tolist() == [True, False, True]


def test_moebius_reconstructs_game(random_game):
    m = moebius_transform(random_game)
    masks = np.arange(len(m))
    recon = np.array([m[(masks & s) == masks].sum() for s in masks])
    np.testing.assert_allclose(recon, random_game, atol=1e-10)


def test_n1_example_from_summary():
    """EU(empty) = EU(age) = EU(experience) = 0.05, EU(both) = 0.40."""
    v = np.array([0.05, 0.05, 0.05, 0.40])
    m = compute_index(v, "Moebius", 2)
    assert m[(0, 1)] == pytest.approx(0.35)
    assert m[(0,)] == pytest.approx(0.0) == pytest.approx(m[(1,)])
    sv = compute_index(v, "SV")
    assert sv[(0,)] == pytest.approx(0.175) == pytest.approx(sv[(1,)])


def test_shapley_matches_brute_force(random_game):
    sv = np.array(list(compute_index(random_game, "SV").values()))
    np.testing.assert_allclose(sv, brute_force_shapley(random_game, 5), atol=1e-10)


@pytest.mark.parametrize("max_order", [1, 2, 3, 5])
def test_k_sii_is_efficient(random_game, max_order):
    k = compute_index(random_game, "k-SII", max_order)
    assert sum(k.values()) == pytest.approx(random_game[-1] - random_game[0])


def test_k_sii_of_full_order_is_moebius(random_game):
    k = compute_index(random_game, "k-SII", 5)
    m = compute_index(random_game, "Moebius", 5)
    for s in k:
        assert k[s] == pytest.approx(m[s])


def test_pairwise_sii_definition(random_game):
    """SII(ij) = sum_S |S|!(n-|S|-2)!/(n-1)! * discrete derivative."""
    n, i, j = 5, 1, 3
    expected = 0.0
    others = [p for p in range(n) if p not in (i, j)]
    for size in range(n - 1):
        w = factorial(size) * factorial(n - size - 2) / factorial(n - 1)
        for s in combinations(others, size):
            mask = sum(1 << p for p in s)
            v = random_game
            expected += w * (v[mask | 1 << i | 1 << j] - v[mask | 1 << i] - v[mask | 1 << j] + v[mask])
    assert compute_index(random_game, "SII", 2)[(i, j)] == pytest.approx(expected)


def test_indices_are_linear(random_game):
    other = np.random.default_rng(1).normal(size=2**5)
    for index in ("SV", "k-SII", "SII", "Moebius"):
        a, b, ab = (compute_index(g, index, 2) for g in (random_game, other, random_game + other))
        for s in ab:
            assert ab[s] == pytest.approx(a[s] + b[s])


def test_permutation_sampling_converges(random_game):
    def game(coalitions):
        idx = (coalitions * (1 << np.arange(5))).sum(axis=1)
        return random_game[idx][:, None]

    phi = permutation_shapley(game, 5, budget=20_000, random_state=0)[:, 0]
    np.testing.assert_allclose(phi, brute_force_shapley(random_game, 5), atol=0.05)
    assert phi.sum() == pytest.approx(random_game[-1] - random_game[0])


def test_unknown_index(random_game):
    with pytest.raises(ValueError, match="Unknown index"):
        compute_index(random_game, "nope")
