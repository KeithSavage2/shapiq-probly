"""Result object returned by every explainer call."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .exact import Interaction, compute_index
from .measures import NOTIONS


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
        predicted_proba: Mean predictive distribution at ``x``.
        predicted_class: Index of the most probable class.
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
    predicted_proba: np.ndarray
    predicted_class: int
    uncertainty: dict[str, float]
    uncertainty_empty: dict[str, float]
    attributions: dict[str, Attribution]
    shapley: dict[str, np.ndarray]
    game_values: dict[str, np.ndarray] | None = None
    config: dict[str, Any] = field(default_factory=dict)

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
        return ", ".join(f"{self.feature_names[i]} ({sv[i]:+.3f})" for i in top)

    def summary(self) -> str:
        """Answer the guiding question in plain text."""
        u = self.uncertainty
        share = u["epistemic"] / u["total"] if u["total"] > 0 else 0.0
        gain = self.config.get("mode") == "information_gain"
        sign = "positive = feature reduces uncertainty" if gain else "positive = feature raises uncertainty"
        lines = [
            f"Prediction: class {self.predicted_class} with p = {self.predicted_proba[self.predicted_class]:.3f}",
            f"Uncertainty at x: total {u['total']:.3f} = aleatoric {u['aleatoric']:.3f} "
            f"+ epistemic {u['epistemic']:.3f} ({share:.0%} epistemic)",
            f"No feature known: total {self.uncertainty_empty['total']:.3f}, "
            f"aleatoric {self.uncertainty_empty['aleatoric']:.3f}, "
            f"epistemic {self.uncertainty_empty['epistemic']:.3f}",
            f"Shapley values ({sign}):",
        ]
        lines += [f"  {notion:<10} {self._drivers(notion)}" for notion in NOTIONS]
        if self.epistemic.max_order >= 2:
            name, value = self.top_interactions("epistemic", 1)[0]
            lines.append(f"Strongest epistemic interaction ({self.epistemic.index}): {name} ({value:+.3f})")
        if u["total"] < 0.1 * np.log(len(self.predicted_proba)):
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
            "predicted_proba": self.predicted_proba.tolist(),
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
        return cls(
            x=np.asarray(data["x"], dtype=float),
            feature_names=list(data["feature_names"]),
            predicted_proba=np.asarray(data["predicted_proba"], dtype=float),
            predicted_class=int(data["predicted_class"]),
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
