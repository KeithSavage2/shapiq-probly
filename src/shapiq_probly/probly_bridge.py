"""Bridge to probly (optional dependency).

NOTE: this is a deliberately thin sketch. probly could not be installed in the
environment this package was developed in, so the exact shape and type of what
``representer(model).represent(x)`` returns is NOT verified here. Everything
else in the package only needs a function ``X -> (M, n, C)``; this module is
the single place that has to know probly.

To finish it in the fork:

1. Call ``rep = representer(method(model)); out = rep.represent(X)`` once and
   inspect ``out`` (type, shape, axis order).
2. Write the conversion to a numpy array of shape ``(M, n, C)`` and pass it as
   ``to_probs``, or adjust ``_default_to_probs``.
3. Add a test that compares ``UncertaintyGame`` at the full coalition with
   ``quantify(rep.represent(x))`` (``.total``, ``.aleatoric``, ``.epistemic``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .adapters import MemberPredictor


def _default_to_probs(representation: Any) -> np.ndarray:
    """Best-effort conversion of a probly sample representation to ``(M, n, C)``.

    Assumes an array-like (or an object with ``.samples`` / ``.tensor``) laid
    out as ``(n, M, C)``, i.e. batch first, members second. Verify this!
    """
    for attr in ("samples", "tensor", "array"):
        if hasattr(representation, attr):
            representation = getattr(representation, attr)
            break
    if hasattr(representation, "detach"):
        representation = representation.detach().cpu().numpy()
    probs = np.asarray(representation, dtype=float)
    if probs.ndim != 3:
        msg = f"Expected a 3D sample representation, got shape {probs.shape}. Pass a custom 'to_probs'."
        raise ValueError(msg)
    return np.moveaxis(probs, 1, 0)


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
