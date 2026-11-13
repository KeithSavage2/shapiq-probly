"""Monte Carlo dropout as a member predictor.

Every dropout mask defines one thinned sub-network. We draw ``n_samples`` masks
once and reuse them for every input, so each mask behaves like one ensemble
member. That matters for the game: with fresh masks per forward pass the
"members" would not be consistent across coalitions, and averaging a member over
imputations (aggregation A) would be meaningless.

This module contains the variant for fitted sklearn MLPs, which runs anywhere.
sklearn does not train with dropout, so masks are only applied at prediction
time; treat it as a testable stand-in. For networks trained with dropout use
:func:`shapiq_probly.torch_bridge.from_torch_dropout`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import MemberPredictor


def _activate(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "identity":
        return z
    if kind == "softmax":
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)
    msg = f"Unsupported activation {kind!r}."
    raise ValueError(msg)


def mc_dropout_sklearn(
    model: Any,
    n_samples: int = 20,
    p: float = 0.1,
    random_state: int | None = 0,
) -> MemberPredictor:
    """Member predictor that applies MC dropout to a fitted sklearn MLP.

    Args:
        model: Fitted ``MLPClassifier`` or single-output ``MLPRegressor``.
        n_samples: Number of dropout masks, i.e. members ``M``.
        p: Dropout probability on the hidden units.
        random_state: Seed for the masks.

    Returns:
        Callable ``X -> (M, n, C)`` for classifiers or ``X -> (M, n)`` for
        regressors. The network does not predict a variance, so for regression
        the members contribute no aleatoric part of their own.
    """
    if not hasattr(model, "coefs_"):
        msg = f"{type(model).__name__} is not a fitted sklearn MLP (no 'coefs_')."
        raise TypeError(msg)
    if not 0 <= p < 1:
        msg = "p must be in [0, 1)."
        raise ValueError(msg)
    weights, biases = model.coefs_, model.intercepts_
    out_kind = model.out_activation_
    classification = hasattr(model, "classes_")
    n_out = weights[-1].shape[1]
    if (classification and out_kind == "logistic" and n_out > 1) or (not classification and n_out > 1):
        msg = "Multi-label classifiers and multi-output regressors are not supported."
        raise ValueError(msg)
    rng = np.random.default_rng(random_state)
    # One mask per member and hidden layer, with inverted-dropout scaling.
    masks = [(rng.random((n_samples, w.shape[1])) >= p) / (1.0 - p) for w in weights[:-1]]

    def predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = []
        for m in range(n_samples):
            a = X
            for layer, (w, b) in enumerate(zip(weights, biases, strict=True)):
                a = a @ w + b
                if layer < len(weights) - 1:
                    a = _activate(a, model.activation) * masks[layer][m]
            a = _activate(a, out_kind)
            if not classification:
                out.append(a[:, 0])
            elif out_kind == "logistic":
                out.append(np.column_stack([1.0 - a[:, 0], a[:, 0]]))
            else:
                out.append(a)
        return np.stack(out)

    predict.task = "classification" if classification else "regression"  # type: ignore[attr-defined]
    return predict
