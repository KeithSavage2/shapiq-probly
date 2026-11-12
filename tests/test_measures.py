import numpy as np
import pytest

from shapiq_probly.measures import entropy, entropy_decomposition, get_decomposition


def test_entropy_known_values():
    assert entropy(np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert entropy(np.array([1.0, 0.0])) == 0.0


def test_decomposition_is_additive_and_nonnegative():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=(7, 20))
    tu, au, eu = entropy_decomposition(probs)
    assert tu.shape == (20,)
    np.testing.assert_allclose(tu, au + eu, atol=1e-12)
    assert (eu >= -1e-12).all()


def test_disagreeing_members_are_purely_epistemic():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    tu, au, eu = entropy_decomposition(probs)
    assert au == pytest.approx(0.0)
    assert tu == pytest.approx(np.log(2)) == pytest.approx(eu)


def test_agreeing_members_are_purely_aleatoric():
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    _, au, eu = entropy_decomposition(probs)
    assert eu == pytest.approx(0.0)
    assert au == pytest.approx(np.log(2))


def test_unknown_measure():
    with pytest.raises(ValueError, match="Unknown measure"):
        get_decomposition("nope")
