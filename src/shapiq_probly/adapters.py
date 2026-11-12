"""Adapters: turn a model into a member predictor.

A *member predictor* is a callable ``f(X) -> probs`` with ``X`` of shape
``(n, d)`` and ``probs`` of shape ``(M, n, C)``: class probabilities of each of
the ``M`` ensemble members (or dropout samples) for each of the ``n`` points.
Everything downstream only depends on this interface, which is what makes the
explainer independent of the uncertainty method.

We deliberately call ``predict_proba`` ourselves: shapiq prefers logits for
classifiers, but entropies need probabilities.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

MemberPredictor = Callable[[np.ndarray], np.ndarray]


def _aligned_proba(member: Any, X: np.ndarray, classes: np.ndarray | None) -> np.ndarray:
    proba = np.asarray(member.predict_proba(X), dtype=float)
    if classes is None or proba.shape[1] == len(classes):
        return proba
    # A bagged member may not have seen every class.
    out = np.zeros((len(X), len(classes)))
    out[:, np.searchsorted(classes, member.classes_)] = proba
    return out


def from_members(members: Sequence[Any], classes: np.ndarray | None = None) -> MemberPredictor:
    """Member predictor from a list of fitted classifiers with ``predict_proba``."""
    members = list(members)
    if not members:
        msg = "Need at least one ensemble member."
        raise ValueError(msg)

    def predict(X: np.ndarray) -> np.ndarray:
        return np.stack([_aligned_proba(m, X, classes) for m in members])

    return predict


def from_sklearn(model: Any) -> MemberPredictor:
    """Member predictor from a fitted sklearn ensemble classifier.

    Supports everything that exposes a flat ``estimators_`` list of classifiers
    (random forests, extra trees, bagging, voting).
    """
    if not hasattr(model, "estimators_"):
        msg = f"{type(model).__name__} has no 'estimators_'. Is it a fitted sklearn ensemble?"
        raise TypeError(msg)
    members = list(model.estimators_)
    if not all(hasattr(m, "predict_proba") for m in members):
        msg = "All ensemble members need 'predict_proba' (boosting ensembles are not supported)."
        raise TypeError(msg)
    classes = np.asarray(model.classes_) if hasattr(model, "classes_") else None
    features = getattr(model, "estimators_features_", None)

    def predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        out = []
        for i, member in enumerate(members):
            Xi = X if features is None else X[:, features[i]]
            out.append(_aligned_proba(member, Xi, classes))
        return np.stack(out)

    return predict


def as_member_predictor(model: Any) -> MemberPredictor:
    """Resolve a model to a member predictor.

    Accepts a callable ``X -> (M, n, C)``, a fitted sklearn ensemble, or a
    list/tuple of fitted classifiers.
    """
    if hasattr(model, "estimators_"):
        return from_sklearn(model)
    if isinstance(model, (list, tuple)):
        return from_members(model)
    if callable(model):
        return model
    msg = (
        f"Cannot adapt {type(model).__name__}. Pass a callable X -> (M, n, C), a fitted sklearn "
        "ensemble, or a list of fitted classifiers."
    )
    raise TypeError(msg)
