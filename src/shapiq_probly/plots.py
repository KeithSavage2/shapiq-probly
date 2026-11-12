"""Plots that explain prediction and uncertainty together (needs matplotlib)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .explanation import UncertaintyExplanation

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = {"aleatoric": "#2a78d6", "epistemic": "#eb6834", "prediction": "#1baf7a", "total": "#52514e"}


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


def _labels(expl: UncertaintyExplanation) -> list[str]:
    return [f"{name} = {value:.3g}" for name, value in zip(expl.feature_names, expl.x, strict=True)]


def _sign_note(expl: UncertaintyExplanation) -> str:
    if expl.config.get("mode") == "information_gain":
        return "information gain (positive = reduces uncertainty)"
    return "contribution to uncertainty (positive = raises it)"


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
    p = expl.predicted_proba[expl.predicted_class]
    ax_p.set_title(f"Prediction: P(class {expl.predicted_class}) = {p:.2f}", loc="left", color=INK, fontsize=12)
    ax_p.set_xlabel(
        f"contribution to P(class {expl.predicted_class}), baseline {expl.prediction.baseline_value:.2f}",
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
        f"Uncertainty: {u['total']:.2f} = {u['aleatoric']:.2f} aleatoric + {u['epistemic']:.2f} epistemic",
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
    placed: list[np.ndarray] = []
    for i in np.argsort(-eu):
        anchor = ax.transData.transform((au[i], eu[i])) * 72.0 / fig.dpi  # in points
        width = 6.5 * len(expl.feature_names[i])
        best, best_gap = offsets[0], -1.0
        for off in offsets:
            centre = anchor + np.array([off[0] + (width / 2 if off[2] == "left" else -width / 2), 2.2 * off[1]])
            gap = min((float(np.abs((centre - q) / (width, 16)).max()) for q in placed), default=np.inf)
            if gap > best_gap:
                best, best_gap, best_centre = off, gap, centre
            if gap >= 1.0:
                break
        placed.append(best_centre)
        ax.annotate(
            expl.feature_names[i],
            (au[i], eu[i]),
            xytext=best[:2],
            textcoords="offset points",
            ha=best[2],
            va=best[3],
            color=INK,
            fontsize=10,
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
    ax.set_title("Uncertainty profile", loc="left", color=INK, fontsize=12)
    return fig


def plot_interaction_network(
    expl: UncertaintyExplanation,
    notion: str = "epistemic",
    *,
    figsize: tuple[float, float] = (6.4, 6.4),
) -> Any:
    """Network of pairwise interactions for one notion (default: epistemic).

    Node area shows the order-1 value, edge width the order-2 value. Solid and
    filled means positive, dashed and hollow means negative.

    Returns:
        The matplotlib figure.
    """
    plt = _plt()
    from matplotlib.lines import Line2D  # noqa: PLC0415

    attribution = expl.attributions[notion]
    pairs = attribution.order(2)
    if not pairs:
        msg = "The explanation has no order-2 interactions (use index='k-SII', max_order=2)."
        raise ValueError(msg)
    d = expl.n_features
    color = COLORS[notion]
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
            f"{expl.feature_names[i]}\n{first[i]:+.2f}",
            ha="center",
            va="center",
            color=INK,
            fontsize=10,
        )
    name, value = expl.top_interactions(notion, 1, min_order=2)[0]
    ax.set_title(
        f"{notion.capitalize()} interactions ({attribution.index})\nstrongest: {name} ({value:+.2f})",
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
