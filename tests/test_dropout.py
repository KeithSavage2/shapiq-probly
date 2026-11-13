import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor

from shapiq_probly import explain_uncertainty, mc_dropout_sklearn


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int), (X[:, 0] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)


@pytest.fixture(scope="module")
def binary_mlp(data):
    X, y, _ = data
    return MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=600, random_state=0).fit(X, y)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("activation", ["relu", "tanh", "logistic"])
def test_without_dropout_every_member_is_the_network(data, activation):
    X, y_binary, y_multi = data
    for y in (y_binary, y_multi):
        mlp = MLPClassifier(hidden_layer_sizes=(8,), activation=activation, max_iter=60, random_state=0).fit(X, y)
        probs = mc_dropout_sklearn(mlp, n_samples=3, p=0.0)(X[:20])
        assert probs.shape == (3, 20, len(mlp.classes_))
        for member in probs:
            np.testing.assert_allclose(member, mlp.predict_proba(X[:20]), atol=1e-10)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_regressor(data):
    X, _, _ = data
    mlp = MLPRegressor(hidden_layer_sizes=(8,), max_iter=60, random_state=0).fit(X, X[:, 0] - X[:, 1])
    predict = mc_dropout_sklearn(mlp, n_samples=4, p=0.0)
    assert predict.task == "regression"
    out = predict(X[:15])
    assert out.shape == (4, 15)
    np.testing.assert_allclose(out[0], mlp.predict(X[:15]), atol=1e-10)


def test_masks_are_fixed_members(data, binary_mlp):
    X = data[0]
    predict = mc_dropout_sklearn(binary_mlp, n_samples=10, p=0.3, random_state=0)
    assert predict.task == "classification"
    full = predict(X[:20])
    np.testing.assert_array_equal(full, predict(X[:20]))  # same masks on every call
    np.testing.assert_allclose(predict(X[:5]), full[:, :5], atol=1e-12)  # and for every row
    np.testing.assert_allclose(full.sum(axis=-1), 1.0)
    assert np.abs(full - full[0]).max() > 1e-3  # members differ
    other_seed = mc_dropout_sklearn(binary_mlp, n_samples=10, p=0.3, random_state=1)(X[:20])
    assert np.abs(full - other_seed).max() > 1e-3


def test_explanation_is_additive_and_has_epistemic_uncertainty(data, binary_mlp):
    X = data[0]
    predict = mc_dropout_sklearn(binary_mlp, n_samples=15, p=0.3)
    expl = explain_uncertainty(predict, X[0], X, n_samples=30)
    assert expl.config["n_members"] == 15
    assert expl.check_additivity()
    assert expl.uncertainty["epistemic"] > 0


def test_input_validation(data, binary_mlp):
    with pytest.raises(TypeError, match="fitted sklearn MLP"):
        mc_dropout_sklearn(object())
    with pytest.raises(ValueError, match="p must be"):
        mc_dropout_sklearn(binary_mlp, p=1.0)
