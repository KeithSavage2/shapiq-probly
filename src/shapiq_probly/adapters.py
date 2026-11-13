"""Adapters: turn a model into a member predictor.

A *member predictor* is a callable ``f(X)`` with ``X`` of shape ``(n, d)`` that
returns the predictions of each of the ``M`` ensemble members (or dropout
samples) for each of the ``n`` points:

* classification: class probabilities, shape ``(M, n, C)``;
* regression: ``(M, n, 2)`` holding mean and variance per member, or ``(M, n)``
  if the members only give point predictions (variance 0).

Everything downstream only depends on this interface, which is what makes the
explainer independent of the uncertainty method. Adapters set a ``task``
attribute (``"classification"`` or ``"regression"``) on the returned function so
that the game can pick the matching uncertainty measure.

We deliberately call ``predict_proba`` ourselves: shapiq prefers logits for
classifiers, but entropies need probabilities.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

MemberPredictor = Callable[[np.ndarray], np.ndarray]

_VARIANCE_CRITERIA = ("squared_error", "friedman_mse")


def _aligned_proba(member: Any, X: np.ndarray, classes: np.ndarray | None) -> np.ndarray:
    proba = np.asarray(member.predict_proba(X), dtype=float)
    if classes is None or proba.shape[1] == len(classes):
        return proba
    # A bagged member may not have seen every class.
    out = np.zeros((len(X), len(classes)))
    out[:, np.searchsorted(classes, member.classes_)] = proba
    return out


def _mean_and_variance(member: Any, X: np.ndarray) -> np.ndarray:
    """Mean and variance of one regressor, shape ``(n, 2)``.

    Regression trees report the target variance in the leaf, regressors with
    ``predict(X, return_std=True)`` their own predictive variance, all others 0.
    """
    tree = getattr(member, "tree_", None)
    if tree is not None and getattr(member, "criterion", None) in _VARIANCE_CRITERIA and member.n_outputs_ == 1:
        leaves = member.apply(X)
        return np.column_stack([tree.value[leaves].reshape(len(X)), tree.impurity[leaves]])
    if "return_std" in inspect.signature(member.predict).parameters:
        mean, std = member.predict(X, return_std=True)
        return np.column_stack([np.ravel(mean), np.ravel(std) ** 2])
    mean = np.asarray(member.predict(X), dtype=float)
    if mean.ndim != 1:
        msg = "Only single-output regressors are supported."
        raise ValueError(msg)
    return np.column_stack([mean, np.zeros(len(X))])


def _is_classifier(members: Sequence[Any]) -> bool:
    if all(hasattr(m, "predict_proba") for m in members):
        return True
    if any(hasattr(m, "predict_proba") for m in members) or any(hasattr(m, "classes_") for m in members):
        msg = "All ensemble members need 'predict_proba' (boosting ensembles are not supported)."
        raise TypeError(msg)
    if not all(hasattr(m, "predict") for m in members):
        msg = "Ensemble members need 'predict_proba' (classification) or 'predict' (regression)."
        raise TypeError(msg)
    return False


def _build(members: list[Any], classes: np.ndarray | None, features: Any) -> MemberPredictor:
    classification = _is_classifier(members)

    def predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        out = []
        for i, member in enumerate(members):
            Xi = X if features is None else X[:, features[i]]
            out.append(_aligned_proba(member, Xi, classes) if classification else _mean_and_variance(member, Xi))
        return np.stack(out)

    predict.task = "classification" if classification else "regression"  # type: ignore[attr-defined]
    return predict


def from_members(members: Sequence[Any], classes: np.ndarray | None = None) -> MemberPredictor:
    """Member predictor from a list of fitted classifiers or regressors."""
    members = list(members)
    if not members:
        msg = "Need at least one ensemble member."
        raise ValueError(msg)
    return _build(members, classes, None)


def from_sklearn(model: Any) -> MemberPredictor:
    """Member predictor from a fitted sklearn ensemble.

    Supports everything that exposes a flat ``estimators_`` list (random
    forests, extra trees, bagging, voting), for classification and regression.
    For regression trees the leaf variance is used as the member's aleatoric
    variance, so use ``min_samples_leaf`` well above 1.
    """
    if not hasattr(model, "estimators_"):
        msg = f"{type(model).__name__} has no 'estimators_'. Is it a fitted sklearn ensemble?"
        raise TypeError(msg)
    if isinstance(model.estimators_, np.ndarray) and model.estimators_.ndim > 1:
        msg = "All ensemble members need 'predict_proba' (boosting ensembles are not supported)."
        raise TypeError(msg)
    members = list(model.estimators_)
    classes = np.asarray(model.classes_) if hasattr(model, "classes_") else None
    return _build(members, classes, getattr(model, "estimators_features_", None))


def as_member_predictor(model: Any) -> MemberPredictor:
    """Resolve a model to a member predictor.

    Accepts a callable ``X -> (M, n, C)`` (or ``(M, n, 2)`` / ``(M, n)`` for
    regression), a fitted sklearn ensemble, or a list/tuple of fitted models.
    """
    if hasattr(model, "estimators_"):
        return from_sklearn(model)
    if isinstance(model, (list, tuple)):
        return from_members(model)
    if callable(model):
        return model
    msg = (
        f"Cannot adapt {type(model).__name__}. Pass a callable X -> (M, n, C), a fitted sklearn "
        "ensemble, or a list of fitted models."
    )
    raise TypeError(msg)
