"""Bridge to probly (optional dependency).

Turns a probly representer into the function ``X -> (M, n, C)`` that the rest
of the package works with. Verified against probly commit ``2729f4f``
(12.11.2026) in ``tests/test_probly_bridge.py``:

* ``representer(ensemble).represent(X)`` returns a sample whose shape is
  ``(n, M)``: batch first, members on ``sample_axis == 1``. Its
  ``.array.probabilities`` has shape ``(n, M, C)``.
* The model must have members. ``representer(RandomForestClassifier)`` gives a
  single distribution per point; use ``probly.method.ensemble.ensemble`` or
  pass the forest directly to this package (see ``adapters.py``).

Only classification is covered. For regression, probly currently returns no
variance decomposition for sklearn models, so use the built-in adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .adapters import MemberPredictor


def _default_to_probs(representation: Any) -> np.ndarray:
    """Convert a probly sample representation to probabilities ``(M, n, C)``.

    Handles probly samples (anything with ``sample_axis`` and
    ``array.probabilities``) and, as a fallback, plain arrays or tensors laid
    out as ``(n, M, C)``.
    """
    sample_axis = getattr(representation, "sample_axis", None)
    inner = getattr(representation, "array", None)
    if sample_axis is not None and hasattr(inner, "probabilities"):
        values, member_axis = inner.probabilities, int(sample_axis)
    else:
        values, member_axis = representation, 1
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    try:
        probs = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as err:
        msg = f"Cannot convert a {type(representation).__name__} to member probabilities. Pass a custom 'to_probs'."
        raise TypeError(msg) from err
    if probs.ndim != 3:
        msg = (
            f"Expected member probabilities with 3 axes, got shape {probs.shape}. "
            "The model probably has no members (e.g. a plain random forest); wrap it with "
            "probly's 'ensemble' or pass a custom 'to_probs'."
        )
        raise ValueError(msg)
    return np.moveaxis(probs, member_axis, 0)


def from_probly(
    representer: Any,
    to_probs: Callable[[Any], np.ndarray] | None = None,
    to_input: Callable[[np.ndarray], Any] | None = None,
) -> MemberPredictor:
    """Member predictor from a probly representer.

    Args:
        representer: Object with ``represent(X)`` (``probly.representer(model)``).
        to_probs: Converts the representation to probabilities ``(M, n, C)``.
        to_input: Converts the numpy input before calling the model, e.g.
            ``lambda X: torch.as_tensor(X, dtype=torch.float32)``.
    """
    convert = to_probs or _default_to_probs

    def predict(X: np.ndarray) -> np.ndarray:
        inputs = X if to_input is None else to_input(X)
        return convert(representer.represent(inputs))

    return predict
