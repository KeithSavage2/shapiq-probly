"""Member predictors for torch models (optional dependency).

Checked in the CI job ``torch`` by ``tests/test_torch_bridge.py``, so far only
with small untrained networks. The tests are skipped when torch is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .adapters import MemberPredictor


def _to_numpy(out: Any, output: str) -> np.ndarray:
    import torch  # noqa: PLC0415

    if output == "logits":
        out = torch.softmax(out, dim=-1)
    elif output not in ("probs", "regression"):
        msg = f"Unknown output {output!r}. Use 'logits', 'probs' or 'regression'."
        raise ValueError(msg)
    out = out.detach().cpu().numpy().astype(float)
    if output == "regression" and out.ndim == 2 and out.shape[1] == 1:
        out = out[:, 0]
    return out


def _forward(model: Any, X: np.ndarray, output: str, batch_size: int) -> np.ndarray:
    import torch  # noqa: PLC0415

    param = next(model.parameters(), None)
    device = param.device if param is not None else "cpu"
    dtype = param.dtype if param is not None else torch.float32
    chunks = []
    for start in range(0, len(X), batch_size):
        batch = torch.as_tensor(np.asarray(X[start : start + batch_size]), dtype=dtype, device=device)
        chunks.append(_to_numpy(model(batch), output))
    return np.concatenate(chunks)


def from_torch_ensemble(models: Sequence[Any], *, output: str = "logits", batch_size: int = 8192) -> MemberPredictor:
    """Member predictor from a list of torch modules (a deep ensemble).

    Args:
        models: The ensemble members.
        output: What the modules return: ``"logits"`` (softmax is applied),
            ``"probs"``, or ``"regression"`` (``(n,)``, ``(n, 1)`` or
            ``(n, 2)`` holding mean and variance).
        batch_size: Rows per forward pass.
    """
    import torch  # noqa: PLC0415

    models = list(models)
    if not models:
        msg = "Need at least one ensemble member."
        raise ValueError(msg)

    def predict(X: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for model in models:
                was_training = model.training
                model.eval()
                try:
                    out.append(_forward(model, X, output, batch_size))
                finally:
                    model.train(was_training)
        return np.stack(out)

    predict.task = "regression" if output == "regression" else "classification"  # type: ignore[attr-defined]
    return predict


def from_torch_dropout(
    model: Any,
    n_samples: int = 20,
    *,
    output: str = "logits",
    random_state: int = 0,
    batch_size: int = 8192,
) -> MemberPredictor:
    """Member predictor that applies MC dropout to a torch module.

    The module is put in eval mode (so ``nn.Dropout`` is the identity and batch
    norm uses its running statistics) and a forward hook on every ``nn.Dropout``
    multiplies its output by a fixed mask. Mask ``m`` is the same for every
    input row and every call, so each mask behaves like one ensemble member.
    Masks are created on the first forward pass, when the shapes are known.

    Only ``torch.nn.Dropout`` layers are handled (not ``Dropout2d``,
    ``AlphaDropout`` or functional dropout).

    Args:
        model: Module containing ``nn.Dropout`` layers.
        n_samples: Number of masks, i.e. members ``M``.
        output: ``"logits"``, ``"probs"`` or ``"regression"`` (see
            :func:`from_torch_ensemble`).
        random_state: Seed for the masks.
        batch_size: Rows per forward pass.
    """
    import torch  # noqa: PLC0415

    layers = [m for m in model.modules() if type(m) is torch.nn.Dropout]
    if not layers:
        msg = "The model has no torch.nn.Dropout layers."
        raise ValueError(msg)
    generator = torch.Generator().manual_seed(random_state)
    masks: dict[tuple[int, int], Any] = {}
    state = {"member": 0}

    def make_hook(layer_index: int) -> Any:
        def hook(module: Any, _inputs: Any, out: Any) -> Any:
            key = (state["member"], layer_index)
            if key not in masks:
                keep = 1.0 - module.p
                masks[key] = (torch.rand(out.shape[1:], generator=generator) < keep).to(torch.float32) / keep
            return out * masks[key].to(device=out.device, dtype=out.dtype)

        return hook

    def predict(X: np.ndarray) -> np.ndarray:
        was_training = model.training
        model.eval()
        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
        out = []
        try:
            with torch.no_grad():
                for member in range(n_samples):
                    state["member"] = member
                    out.append(_forward(model, X, output, batch_size))
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        return np.stack(out)

    predict.task = "regression" if output == "regression" else "classification"  # type: ignore[attr-defined]
    return predict
