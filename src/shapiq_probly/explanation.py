"""Result object returned by every explainer call."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .exact import Interaction, compute_index
from .measures import NOTIONS


def _signed(value: float) -> str:
    text = format_number(value)
    return text if text.startswith("-") else f"+{text}"


def format_number(value: float) -> str:
    """Compact number for text and plot titles: ``6,660`` instead of ``6.66e+03``."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}"


@dataclass
class Attribution:
    """Interaction values of one game (similar in spirit to ``shapiq.InteractionValues``).

    Attributes:
        notion: ``"total"``, ``"aleatoric"``, ``"epistemic"`` or ``"prediction"``.
        index: Interaction index, e.g. ``"k-SII"``.
        max_order: Largest interaction size contained.
        values: Mapping interaction -> value.
        baseline_value: ``v(empty)``.
        full_value: ``v(N)``.
        estimated: Whether the values were approximated.
    """

    notion: str
    index: str
    max_order: int
    values: dict[Interaction, float]
    baseline_value: float
    full_value: float
    estimated: bool = False

    def __getitem__(self, interaction: int | Interaction) -> float:
        key = (interaction,) if isinstance(interaction, int) else tuple(sorted(interaction))
        return self.values[key]

    def order(self, k: int) -> dict[Interaction, float]:
        """All interactions of size ``k``."""
        return {s: v for s, v in self.values.items() if len(s) == k}

    def first_order(self, n_players: int) -> np.ndarray:
        """Order-1 values as an array of length ``n_players``."""
        return np.array([self.values[(i,)] for i in range(n_players)])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        out = {
            k: getattr(self, k) for k in ("notion", "index", "max_order", "baseline_value", "full_value", "estimated")
        }
        out["values"] = {",".join(map(str, s)): v for s, v in self.values.items()}
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attribution:
        """Inverse of :meth:`to_dict`."""
        data = dict(data)
        data["values"] = {tuple(int(i) for i in k.split(",")): float(v) for k, v in data["values"].items()}
        return cls(**data)


@dataclass
class UncertaintyExplanation:
    """Explanation of one prediction and of its uncertainty.

    Attributes:
        x: The explained point.
        feature_names: One name per feature.
        predicted_proba: Mean predictive distribution at ``x`` (classification only).
        predicted_class: Index of the most probable class (classification only).
        task: ``"classification"`` or ``"regression"``.
        prediction_value: What the prediction game explains: the probability of
            the predicted class, or the mean prediction for regression.
        uncertainty: Total, aleatoric and epistemic uncertainty at ``x``.
        uncertainty_empty: The same when no feature is known.
        attributions: One :class:`Attribution` per game.
        shapley: Shapley values per game, shape ``(n_features,)`` each.
        game_values: All ``2^n`` coalition values per game (bitmask order), or
            ``None`` if the explanation was approximated.
        config: The settings used (imputer, aggregation, mode, ...).
    """

    x: np.ndarray
    feature_names: list[str]
    predicted_proba: np.ndarray | None
    predicted_class: int | None
    uncertainty: dict[str, float]
    uncertainty_empty: dict[str, float]
    attributions: dict[str, Attribution]
    shapley: dict[str, np.ndarray]
    game_values: dict[str, np.ndarray] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    task: str = "classification"
    prediction_value: float | None = None

    def __post_init__(self) -> None:
        if self.prediction_value is None and self.predicted_proba is not None:
            self.prediction_value = float(self.predicted_proba[self.predicted_class])

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.feature_names)

    @property
    def prediction(self) -> Attribution:
        """Attribution of the predicted-class probability."""
        return self.attributions["prediction"]

    @property
    def total(self) -> Attribution:
        """Attribution of the total uncertainty."""
        return self.attributions["total"]

    @property
    def aleatoric(self) -> Attribution:
        """Attribution of the aleatoric uncertainty."""
        return self.attributions["aleatoric"]

    @property
    def epistemic(self) -> Attribution:
        """Attribution of the epistemic uncertainty."""
        return self.attributions["epistemic"]

    def check_additivity(self, atol: float = 1e-8) -> bool:
        """Whether ``phi(TU) = phi(AU) + phi(EU)`` holds for every interaction."""
        t, a, e = self.total.values, self.aleatoric.values, self.epistemic.values
        ok = all(abs(t[s] - a[s] - e[s]) <= atol for s in t)
        sv = self.shapley
        return ok and bool(np.allclose(sv["total"], sv["aleatoric"] + sv["epistemic"], atol=atol))

    def interaction_name(self, interaction: Interaction) -> str:
        """Readable name such as ``"age x experience"``."""
        return " x ".join(self.feature_names[i] for i in interaction)

    def top_interactions(self, notion: str = "epistemic", k: int = 5, min_order: int = 2) -> list[tuple[str, float]]:
        """The ``k`` interactions with the largest absolute value."""
        items = [(s, v) for s, v in self.attributions[notion].values.items() if len(s) >= min_order]
        items.sort(key=lambda sv: -abs(sv[1]))
        return [(self.interaction_name(s), v) for s, v in items[:k]]

    def moebius(self, notion: str = "epistemic", max_order: int = 2) -> dict[Interaction, float]:
        """Moebius coefficients (diagnostic view; amplifies imputation noise)."""
        if self.game_values is None:
            msg = "Moebius coefficients need the exact game values (not available after approximation)."
            raise ValueError(msg)
        return compute_index(self.game_values[notion], "Moebius", max_order)

    def _drivers(self, notion: str, k: int = 3) -> str:
        sv = self.shapley[notion]
        top = np.argsort(-np.abs(sv))[:k]
        return ", ".join(f"{self.feature_names[i]} ({_signed(sv[i])})" for i in top)

    @property
    def prediction_label(self) -> str:
        """Short description of the prediction, e.g. ``"P(class 1) = 0.93"``."""
        if self.task == "regression":
            return f"{self.prediction_value:.4g}"
        return f"P(class {self.predicted_class}) = {self.prediction_value:.2f}"

    def _is_confident(self) -> bool:
        total = self.uncertainty["total"]
        if self.task == "regression":
            return total < 0.1 * self.uncertainty_empty["total"]
        return total < 0.1 * np.log(len(self.predicted_proba))

    def summary(self) -> str:
        """Answer the guiding question in plain text."""
        u = self.uncertainty
        share = u["epistemic"] / u["total"] if u["total"] > 0 else 0.0
        gain = self.config.get("mode") == "information_gain"
        sign = "positive = feature reduces uncertainty" if gain else "positive = feature raises uncertainty"
        unit = f" [{self.config['unit']}]" if self.config.get("unit") else ""
        if self.task == "regression":
            head = f"Prediction: {self.prediction_value:.4g} (standard deviation {np.sqrt(max(u['total'], 0)):.3g})"
        else:
            head = f"Prediction: class {self.predicted_class} with p = {self.prediction_value:.3f}"
        num, empty = format_number, self.uncertainty_empty
        lines = [
            head,
            f"Uncertainty at x{unit}: total {num(u['total'])} = aleatoric {num(u['aleatoric'])} "
            f"+ epistemic {num(u['epistemic'])} ({share:.0%} epistemic)",
            f"No feature known: total {num(empty['total'])}, aleatoric {num(empty['aleatoric'])}, "
            f"epistemic {num(empty['epistemic'])}",
            f"Shapley values ({sign}):",
        ]
        lines += [f"  {notion:<10} {self._drivers(notion)}" for notion in NOTIONS]
        if self.epistemic.max_order >= 2:
            name, value = self.top_interactions("epistemic", 1)[0]
            lines.append(f"Strongest epistemic interaction ({self.epistemic.index}): {name} ({_signed(value)})")
        if self._is_confident():
            lines.append("Reading: the model is confident here; there is little uncertainty to explain.")
        elif share >= 0.5:
            lines.append("Reading: mostly missing knowledge. More data or measuring further features can reduce it.")
        else:
            lines.append("Reading: mostly noise in the data. It cannot be reduced by collecting more data.")
        if self.total.estimated:
            lines.append("Note: values are approximated (too many features for exact computation).")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        return {
            "x": self.x.tolist(),
            "feature_names": list(self.feature_names),
            "task": self.task,
            "prediction_value": self.prediction_value,
            "predicted_proba": None if self.predicted_proba is None else self.predicted_proba.tolist(),
            "predicted_class": self.predicted_class,
            "uncertainty": self.uncertainty,
            "uncertainty_empty": self.uncertainty_empty,
            "attributions": {k: a.to_dict() for k, a in self.attributions.items()},
            "shapley": {k: v.tolist() for k, v in self.shapley.items()},
            "game_values": None if self.game_values is None else {k: v.tolist() for k, v in self.game_values.items()},
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UncertaintyExplanation:
        """Inverse of :meth:`to_dict`."""
        gv = data.get("game_values")
        proba, cls_index = data.get("predicted_proba"), data.get("predicted_class")
        return cls(
            x=np.asarray(data["x"], dtype=float),
            feature_names=list(data["feature_names"]),
            predicted_proba=None if proba is None else np.asarray(proba, dtype=float),
            predicted_class=None if cls_index is None else int(cls_index),
            task=data.get("task", "classification"),
            prediction_value=data.get("prediction_value"),
            uncertainty=dict(data["uncertainty"]),
            uncertainty_empty=dict(data["uncertainty_empty"]),
            attributions={k: Attribution.from_dict(a) for k, a in data["attributions"].items()},
            shapley={k: np.asarray(v, dtype=float) for k, v in data["shapley"].items()},
            game_values=None if gv is None else {k: np.asarray(v, dtype=float) for k, v in gv.items()},
            config=dict(data.get("config", {})),
        )

    def save(self, path: str | Path) -> None:
        """Write the explanation to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> UncertaintyExplanation:
        """Read an explanation from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def global_importance(explanations: Sequence[UncertaintyExplanation]) -> dict[str, np.ndarray]:
    """Mean absolute Shapley value per feature and game over several explanations.

    Args:
        explanations: Explanations of several points, e.g. from ``explain_many``.

    Returns:
        Mapping from game name (``"total"``, ``"aleatoric"``, ``"epistemic"``,
        ``"prediction"``) to an array of shape ``(n_features,)``.
    """
    explanations = list(explanations)
    if not explanations:
        msg = "Need at least one explanation."
        raise ValueError(msg)
    return {name: np.mean([np.abs(e.shapley[name]) for e in explanations], axis=0) for name in explanations[0].shapley}


@dataclass
class ScalarExplanation:
    """Explanation of a single quantity, e.g. a conformal set size or an OOD score.

    Attributes:
        name: Name of the explained quantity.
        x: The explained point.
        feature_names: One name per feature.
        value: The quantity at ``x`` (all features known).
        baseline: The quantity when no feature is known.
        attribution: Interaction values.
        shapley: Shapley values, shape ``(n_features,)``.
        game_values: All ``2^n`` coalition values (bitmask order), or ``None``
            if the explanation was approximated.
        info: Extra facts about the game (e.g. the conformal threshold).
    """

    name: str
    x: np.ndarray
    feature_names: list[str]
    value: float
    baseline: float
    attribution: Attribution
    shapley: np.ndarray
    game_values: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.feature_names)

    def interaction_name(self, interaction: Interaction) -> str:
        """Readable name such as ``"age x experience"``."""
        return " x ".join(self.feature_names[i] for i in interaction)

    def top_interactions(self, k: int = 5, min_order: int = 2) -> list[tuple[str, float]]:
        """The ``k`` interactions with the largest absolute value."""
        items = [(s, v) for s, v in self.attribution.values.items() if len(s) >= min_order]
        items.sort(key=lambda sv: -abs(sv[1]))
        return [(self.interaction_name(s), v) for s, v in items[:k]]

    def summary(self) -> str:
        """Plain-text summary of the explanation."""
        top = np.argsort(-np.abs(self.shapley))[:3]
        drivers = ", ".join(f"{self.feature_names[i]} ({_signed(self.shapley[i])})" for i in top)
        lines = [
            f"{self.name} at x: {format_number(self.value)} (no feature known: {format_number(self.baseline)})",
            f"Shapley values (positive = feature raises the {self.name}): {drivers}",
        ]
        if self.attribution.max_order >= 2:
            name, value = self.top_interactions(1)[0]
            lines.append(f"Strongest interaction ({self.attribution.index}): {name} ({_signed(value)})")
        lines += [f"{key}: {val}" for key, val in self.info.items()]
        if self.attribution.estimated:
            lines.append("Note: values are approximated (too many features for exact computation).")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "x": self.x.tolist(),
            "feature_names": list(self.feature_names),
            "value": self.value,
            "baseline": self.baseline,
            "attribution": self.attribution.to_dict(),
            "shapley": self.shapley.tolist(),
            "game_values": None if self.game_values is None else self.game_values.tolist(),
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScalarExplanation:
        """Inverse of :meth:`to_dict`."""
        gv = data.get("game_values")
        return cls(
            name=data["name"],
            x=np.asarray(data["x"], dtype=float),
            feature_names=list(data["feature_names"]),
            value=float(data["value"]),
            baseline=float(data["baseline"]),
            attribution=Attribution.from_dict(data["attribution"]),
            shapley=np.asarray(data["shapley"], dtype=float),
            game_values=None if gv is None else np.asarray(gv, dtype=float),
            info=dict(data.get("info", {})),
        )

    def save(self, path: str | Path) -> None:
        """Write the explanation to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ScalarExplanation:
        """Read an explanation from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))
