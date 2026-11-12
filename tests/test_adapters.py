import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from shapiq_probly import as_member_predictor, from_members


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1).astype(int)
    return X, y


def test_random_forest_members_average_to_predict_proba(data):
    X, y = data
    model = RandomForestClassifier(n_estimators=15, random_state=0).fit(X, y)
    probs = as_member_predictor(model)(X[:20])
    assert probs.shape == (15, 20, 3)
    np.testing.assert_allclose(probs.mean(axis=0), model.predict_proba(X[:20]), atol=1e-12)


def test_bagging_with_feature_subsets_and_missing_classes(data):
    X, y = data
    y = y.copy()
    y[5:] = np.minimum(y[5:], 1)  # class 2 is rare, some members never see it
    y[:5] = 2
    model = BaggingClassifier(
        DecisionTreeClassifier(max_depth=3), n_estimators=12, max_features=0.5, max_samples=0.3, random_state=0
    ).fit(X, y)
    probs = as_member_predictor(model)(X[:10])
    assert probs.shape == (12, 10, 3)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)
    np.testing.assert_allclose(probs.mean(axis=0), model.predict_proba(X[:10]), atol=1e-12)


def test_list_of_models(data):
    X, y = data
    members = [LogisticRegression(C=c).fit(X, y) for c in (0.1, 1.0, 10.0)]
    assert as_member_predictor(members)(X[:7]).shape == (3, 7, 3)
    with pytest.raises(ValueError, match="at least one"):
        from_members([])


def test_unsupported_models(data):
    X, y = data
    with pytest.raises(TypeError, match="predict_proba"):
        as_member_predictor(GradientBoostingClassifier(n_estimators=3).fit(X, y))
    with pytest.raises(TypeError, match="Cannot adapt"):
        as_member_predictor(42)
