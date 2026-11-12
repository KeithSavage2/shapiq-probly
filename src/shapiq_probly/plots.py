"""Plots that explain prediction and uncertainty together (needs matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .explanation import ScalarExplanation, format_number, global_importance

if TYPE_CHECKING:
    from .explanation import UncertaintyExplanation

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = {"aleatoric": "#2a78d6", "epistemic": "#eb6834", "prediction": "#1baf7a", "total": "#52514e"}
# One-hue ramps (light = low, dark = high) for encoding the feature value.
RAMPS = {
    "aleatoric": ("#86b6ef", "#2a78d6", "#0d366b"),
    "epistemic": ("#f4b79c", "#eb6834", "#7d2c0e"),
    "total": ("#c3c2b7", "#898781", "#0b0b0b"),
    "prediction": ("#9adfc4", "#1baf7a", "#0a5c40"),
}


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover
        msg = "Plotting needs matplotlib: pip install shapiq-probly[plots]"
        raise ImportError(msg) from err
    return plt


def _style(ax: Any, *, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _short(value: float) -> str:
    return format_number(value) if abs(value) >= 1000 else f"{value:.2f}"


def _signed_short(value: float) -> str:
    """Signed number without exponent notation: ``+842``, ``-0.29``, ``+0.013``."""
    if abs(value) >= 1000:
        text = f"{value:,.0f}"
    elif abs(value) >= 10:
        text = f"{value:.0f}"
    else:
        text = f"{value:.3f}" if abs(value) < 0.1 else f"{value:.2f}"
    if float(text.replace(",", "")) == 0:
        return "0"
    return text if text.startswith("-") else f"+{text}"


def _labels(expl: UncertaintyExplanation) -> list[str]:
    return [f"{name} = {value:.3g}" for name, value in zip(expl.feature_names, expl.x, strict=True)]


def _sign_note(expl: UncertaintyExplanation) -> str:
    unit = f" in {expl.config['unit']}" if expl.config.get("unit") else ""
    if expl.config.get("mode") == "information_gain":
        return f"information gain{unit} (positive = reduces uncertainty)"
    return f"contribution to uncertainty{unit} (positive = raises it)"


def plot_uncertainty_explanation(expl: UncertaintyExplanation, *, figsize: tuple[float, float] | None = None) -> Any:
    """Shapley values of the prediction next to those of AU and EU.

    Left: which features push the probability of the predicted class.
    Right: which features drive aleatoric and epistemic uncertainty; the black
    tick marks their sum, the contribution to total uncertainty.

    Returns:
        The matplotlib figure.
    """
    plt = _plt()
    d = expl.n_features
    order = np.argsort(np.abs(expl.shapley["total"]))  # largest on top
    y = np.arange(d)
    labels = [_labels(expl)[i] for i in order]
    fig, (ax_p, ax_u) = plt.subplots(
        1, 2, figsize=figsize or (11, 1.6 + 0.55 * d), sharey=True, facecolor=SURFACE, layout="constrained"
    )

    pred = expl.shapley["prediction"][order]
    ax_p.barh(y, pred, height=0.5, color=COLORS["prediction"])
    ax_p.set_yticks(y, labels)
    ax_p.tick_params(axis="y", labelcolor=INK, labelsize=10)
    target = "the prediction" if expl.task == "regression" else f"P(class {expl.predicted_class})"
    ax_p.set_title(f"Prediction: {expl.prediction_label}", loc="left", color=INK, fontsize=12)
    ax_p.set_xlabel(
        f"contribution to {target}, baseline {expl.prediction.baseline_value:.3g}",
        color=INK_SECONDARY,
        fontsize=9,
    )

    h = 0.3
    au, eu, tu = (expl.shapley[k][order] for k in ("aleatoric", "epistemic", "total"))
    ax_u.barh(y + h / 2 + 0.02, au, height=h, color=COLORS["aleatoric"], label="aleatoric (noise)")
    ax_u.barh(y - h / 2 - 0.02, eu, height=h, color=COLORS["epistemic"], label="epistemic (missing knowledge)")
    ax_u.scatter(tu, y, marker="|", s=420, color=INK, linewidths=1.6, zorder=3, label="total = sum")
    u = expl.uncertainty
    ax_u.set_title(
        f"Uncertainty: {_short(u['total'])} = {_short(u['aleatoric'])} aleatoric + {_short(u['epistemic'])} epistemic",
        loc="left",
        color=INK,
        fontsize=12,
    )
    ax_u.set_xlabel(_sign_note(expl), color=INK_SECONDARY, fontsize=9)
    fig.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="outside lower right", ncols=3)

    for ax in (ax_p, ax_u):
        _style(ax)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.spines["left"].set_visible(False)
    return fig


def plot_uncertainty_profile(expl: UncertaintyExplanation, *, figsize: tuple[float, float] = (6.4, 6.0)) -> Any:
    """Uncertainty profile: aleatoric (x) against epistemic (y) Shapley value per feature.

    The four quadrants answer "noise or missing knowledge?" for every feature.

    Returns:
        The matplotlib figure.
    """
    plt = _plt()
    au, eu = expl.shapley["aleatoric"], expl.shapley["epistemic"]
    lim = 1.25 * max(np.abs(au).max(), np.abs(eu).max(), 1e-6)
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE, layout="constrained")
    _style(ax, grid_axis="both")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.scatter(au, eu, s=70, color=COLORS["total"], edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # Place labels greedily so that close points do not get overlapping names.
    offsets = [(8, 7, "left", "bottom"), (8, -7, "left", "top"), (-8, 7, "right", "bottom"), (-8, -7, "right", "top")]
    offsets += [(2.6 * dx, 2.6 * dy, ha, va) for dx, dy, ha, va in offsets]
    anchors = ax.transData.transform(np.column_stack([au, eu])) * 72.0 / fig.dpi  # in points
    placed: list[tuple[np.ndarray, float]] = [(a, 12.0) for a in anchors]  # keep labels off the dots too
    # With many features the ones at the origin only clutter the plot: name them in a note instead.
    radius = np.hypot(au, eu)
    labelled = radius >= (0.05 * lim if expl.n_features > 6 else 0.0)
    if not labelled.all():
        rest = ", ".join(expl.feature_names[i] for i in np.argsort(-radius) if not labelled[i])
        ax.text(0, 1.012, f"near zero, not labelled: {rest}", transform=ax.transAxes, color=MUTED, fontsize=8.5)
    for i in np.argsort(-eu):
        if not labelled[i]:
            continue
        anchor = anchors[i]
        width = 6.5 * len(expl.feature_names[i])
        best, best_gap = offsets[0], -1.0
        for off in offsets:
            centre = anchor + np.array(
                [off[0] + (width / 2 if off[2] == "left" else -width / 2), off[1] + np.sign(off[1]) * 7]
            )
            gap = min(float(np.abs((centre - q) / ((width + w) / 2, 14)).max()) for q, w in placed)
            if gap > best_gap:
                best, best_gap, best_centre = off, gap, centre
            if gap >= 1.0:
                break
        placed.append((best_centre, width))
        moved_away = abs(best[0]) > 8
        ax.annotate(
            expl.feature_names[i],
            (au[i], eu[i]),
            xytext=best[:2],
            textcoords="offset points",
            ha=best[2],
            va=best[3],
            color=INK,
            fontsize=10,
            # A label that had to move away from its dot gets a thin leader line.
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7, "shrinkB": 5} if moved_away else None,
        )
    gain = expl.config.get("mode") == "information_gain"
    more, less = ("resolves", "adds") if gain else ("adds", "removes")
    corners = {
        (1, 1): f"{more} noise and\n{more} knowledge gap",
        (-1, 1): f"{less} noise,\n{more} knowledge gap",
        (-1, -1): f"{less} noise and\n{less} knowledge gap",
        (1, -1): f"{more} noise,\n{less} knowledge gap",
    }
    for (sx, sy), text in corners.items():
        ax.text(
            0.96 * sx * lim,
            0.96 * sy * lim,
            text,
            ha="right" if sx > 0 else "left",
            va="top" if sy > 0 else "bottom",
            color=MUTED,
            fontsize=8.5,
        )
    ax.set_xlabel("aleatoric Shapley value (noise)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("epistemic Shapley value (missing knowledge)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Uncertainty profile", loc="left", color=INK, fontsize=12, pad=6 if labelled.all() else 20)
    return fig


def plot_interaction_network(
    expl: UncertaintyExplanation | ScalarExplanation,
    notion: str = "epistemic",
    *,
    figsize: tuple[float, float] = (6.4, 6.4),
) -> Any:
    """Network of pairwise interactions for one notion (default: epistemic).

    Node area shows the order-1 value, edge width the order-2 value. Solid and
    filled means positive, dashed and hollow means negative. For a
    :class:`~shapiq_probly.explanation.ScalarExplanation` ``notion`` is ignored.

    Returns:
        The matplotlib figure.
    """
    plt = _plt()
    from matplotlib.lines import Line2D  # noqa: PLC0415

    scalar = isinstance(expl, ScalarExplanation)
    attribution = expl.attribution if scalar else expl.attributions[notion]
    label = expl.name if scalar else notion
    pairs = attribution.order(2)
    if not pairs:
        msg = "The explanation has no order-2 interactions (use index='k-SII', max_order=2)."
        raise ValueError(msg)
    d = expl.n_features
    color = COLORS["total"] if scalar else COLORS[notion]
    angles = np.pi / 2 - 2 * np.pi * np.arange(d) / d
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    first = attribution.first_order(d)
    scale = max(np.abs(first).max(), max(abs(v) for v in pairs.values()), 1e-12)

    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    for (i, j), value in pairs.items():
        width = 9 * abs(value) / scale
        if width < 0.15:
            continue
        ax.plot(
            pos[[i, j], 0],
            pos[[i, j], 1],
            color=color,
            linewidth=width,
            linestyle="-" if value > 0 else (0, (2, 1.2)),
            alpha=0.85,
            solid_capstyle="round",
            zorder=1,
        )
    for i in range(d):
        size = 150 + 1500 * abs(first[i]) / scale
        face = color if first[i] > 0 else SURFACE
        ax.scatter(*pos[i], s=size, facecolor=face, edgecolors=color, linewidths=2, zorder=2)
        ax.text(
            *(1.28 * pos[i]),
            f"{expl.feature_names[i]}\n{_signed_short(first[i])}",
            ha="center",
            va="center",
            color=INK,
            fontsize=10,
        )
    name, value = max(((expl.interaction_name(pair), v) for pair, v in pairs.items()), key=lambda nv: abs(nv[1]))
    ax.set_title(
        f"{label[:1].upper()}{label[1:]} interactions ({attribution.index})\n"
        f"strongest: {name} ({_signed_short(value)})",
        loc="left",
        color=INK,
        fontsize=12,
    )
    handles = [
        Line2D([], [], color=color, linewidth=3, label="positive"),
        Line2D([], [], color=color, linewidth=3, linestyle=(0, (2, 1.2)), label="negative"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def plot_scalar_explanation(expl: ScalarExplanation, *, figsize: tuple[float, float] | None = None) -> Any:
    """Shapley values of a single quantity (conformal set size, OOD score, ...).

    Returns:
        The matplotlib figure.
    """
    plt = _plt()
    d = expl.n_features
    order = np.argsort(np.abs(expl.shapley))  # largest on top
    labels = [f"{expl.feature_names[i]} = {expl.x[i]:.3g}" for i in order]
    fig, ax = plt.subplots(figsize=figsize or (6.4, 1.6 + 0.5 * d), facecolor=SURFACE, layout="constrained")
    ax.barh(np.arange(d), expl.shapley[order], height=0.5, color=COLORS["total"])
    ax.set_yticks(np.arange(d), labels)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=10)
    _style(ax)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{expl.name[:1].upper()}{expl.name[1:]}: {expl.value:.3g}", loc="left", color=INK, fontsize=12)
    ax.set_xlabel(f"contribution to the {expl.name}, baseline {expl.baseline:.3g}", color=INK_SECONDARY, fontsize=9)
    return fig


def _swarm_offsets(values: np.ndarray, lo: float, hi: float, n_bins: int = 70, step: float = 0.07) -> np.ndarray:
    """Vertical offsets that spread points with similar values (a simple beeswarm)."""
    bins = np.floor((values - lo) / max(hi - lo, 1e-12) * n_bins).astype(int)
    offsets = np.zeros(len(values))
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        k = np.arange(len(idx))
        offsets[idx] = ((k + 1) // 2) * np.where(k % 2 == 1, 1.0, -1.0)
    peak = np.abs(offsets).max()
    return offsets * min(step, 0.4 / peak) if peak > 0 else offsets


def plot_uncertainty_beeswarm(
    explanations: Sequence[UncertaintyExplanation],
    notions: Sequence[str] = ("aleatoric", "epistemic"),
    *,
    max_features: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Global view: Shapley values of many explained points, one panel per notion.

    Every dot is one explained point. Its position shows the Shapley value of
    the feature for that point, its shade the feature value (light = low,
    dark = high, ranked within the feature). Panels share the x-axis so that
    aleatoric and epistemic contributions can be compared directly.

    Args:
        explanations: Explanations of several points, e.g. from ``explain_many``.
        notions: Which games to show.
        max_features: Show only the most important features.
        figsize: Figure size.

    Returns:
        The matplotlib figure.
    """
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415

    explanations = list(explanations)
    first = explanations[0]
    importance = global_importance(explanations)
    combined = sum(importance[n] for n in notions)
    order = np.argsort(-combined)[: max_features or first.n_features][::-1]  # largest on top
    X = np.array([e.x for e in explanations])
    ranks = X.argsort(axis=0).argsort(axis=0) / max(len(explanations) - 1, 1)
    values = {n: np.array([e.shapley[n] for e in explanations]) for n in notions}
    lim = 1.08 * max(np.abs(v[:, order]).max() for v in values.values())
    lim = max(lim, 1e-9)

    fig, axes = plt.subplots(
        1,
        len(notions),
        figsize=figsize or (5.2 * len(notions), 1.9 + 0.55 * len(order)),
        sharex=True,
        sharey=True,
        facecolor=SURFACE,
        layout="constrained",
        squeeze=False,
    )
    for ax, notion in zip(axes[0], notions, strict=True):
        cmap = LinearSegmentedColormap.from_list(notion, RAMPS[notion])
        for row, feature in enumerate(order):
            x = values[notion][:, feature]
            y = row + _swarm_offsets(x, -lim, lim)
            mappable = ax.scatter(
                x, y, c=ranks[:, feature], cmap=cmap, vmin=0, vmax=1, s=16, linewidths=0.4, edgecolors=SURFACE
            )
        _style(ax)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-lim, lim)
        ax.set_title(f"{notion.capitalize()} uncertainty", loc="left", color=INK, fontsize=12)
        ax.set_xlabel(_sign_note(first), color=INK_SECONDARY, fontsize=9)
        bar = fig.colorbar(mappable, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, aspect=40, ticks=[0, 1])
        bar.ax.set_xticklabels(["low", "high"], color=MUTED, fontsize=8)
        bar.set_label("feature value (rank within feature)", color=INK_SECONDARY, fontsize=8)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0)
    axes[0][0].set_yticks(np.arange(len(order)), [first.feature_names[i] for i in order])
    axes[0][0].tick_params(axis="y", labelcolor=INK, labelsize=10)
    fig.suptitle(f"{len(explanations)} explained points", x=0.01, ha="left", color=INK_SECONDARY, fontsize=9)
    return fig


def plot_global_importance(
    explanations: Sequence[UncertaintyExplanation],
    *,
    max_features: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Global importance: mean absolute Shapley value per feature, aleatoric next to epistemic.

    Returns:
        The matplotlib figure.
    """
    plt = _plt()
    explanations = list(explanations)
    first = explanations[0]
    importance = global_importance(explanations)
    combined = importance["aleatoric"] + importance["epistemic"]
    order = np.argsort(-combined)[: max_features or first.n_features][::-1]
    y = np.arange(len(order))
    h = 0.3
    fig, ax = plt.subplots(figsize=figsize or (6.8, 1.6 + 0.55 * len(order)), facecolor=SURFACE, layout="constrained")
    ax.barh(y + h / 2 + 0.02, importance["aleatoric"][order], height=h, color=COLORS["aleatoric"], label="aleatoric")
    ax.barh(y - h / 2 - 0.02, importance["epistemic"][order], height=h, color=COLORS["epistemic"], label="epistemic")
    ax.set_yticks(y, [first.feature_names[i] for i in order])
    ax.tick_params(axis="y", labelcolor=INK, labelsize=10)
    _style(ax)
    ax.spines["left"].set_color(AXIS)
    unit = f" in {first.config['unit']}" if first.config.get("unit") else ""
    ax.set_xlabel(f"mean |Shapley value|{unit}", color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        f"Global uncertainty drivers ({len(explanations)} explained points)", loc="left", color=INK, fontsize=12
    )
    fig.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="outside lower right", ncols=2)
    return fig
