"""Quickstart on the synthetic benchmark: explain three points with known ground truth.

Run from the repository root:  python examples/quickstart.py
Figures are written to examples/output/.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from sklearn.ensemble import RandomForestClassifier

from shapiq_probly import (
    explain_uncertainty,
    make_uncertainty_benchmark,
    plot_interaction_network,
    plot_uncertainty_explanation,
    plot_uncertainty_profile,
)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

data = make_uncertainty_benchmark(n_samples=4000, random_state=0)
# min_samples_leaf > 1 so that single trees output probabilities instead of 0/1.
forest = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, random_state=0).fit(data.X, data.y)

for name, x in data.queries.items():
    expl = explain_uncertainty(forest, x, background=data.X, imputer="knn", feature_names=list(data.feature_names))
    print(f"\n=== query '{name}' ===")
    print(expl.summary())
    print("additive:", expl.check_additivity())
    plot_uncertainty_explanation(expl).savefig(OUT / f"{name}_explanation.png", dpi=150)
    plot_uncertainty_profile(expl).savefig(OUT / f"{name}_profile.png", dpi=150)
    plot_interaction_network(expl).savefig(OUT / f"{name}_network.png", dpi=150)
    expl.save(OUT / f"{name}.json")

print(f"\nFigures written to {OUT}/")
