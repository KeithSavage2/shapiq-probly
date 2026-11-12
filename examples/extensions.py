"""Tour of the extensions on the synthetic benchmark (known ground truth).

1. Regression (variance decomposition)
2. MC dropout (sklearn MLP with fixed dropout masks)
3. Credal sets (upper minus lower entropy)
4. Conformal prediction (size of the prediction set)
5. Out-of-distribution score (kNN distance)
6. Global view (beeswarm over many points)

Run from the repository root:  python examples/extensions.py
Figures are written to examples/output/.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier

from shapiq_probly import (
    ConformalSetSizeGame,
    UncertaintyExplainer,
    explain_ood,
    explain_set_size,
    explain_uncertainty,
    make_uncertainty_benchmark,
    mc_dropout_sklearn,
    plot_global_importance,
    plot_interaction_network,
    plot_scalar_explanation,
    plot_uncertainty_beeswarm,
    plot_uncertainty_explanation,
)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


data = make_uncertainty_benchmark(n_samples=6000, random_state=0)
names = list(data.feature_names)
X_train, y_train = data.X[:4000], data.y[:4000]
X_cal, y_cal = data.X[4000:5000], data.y[4000:5000]
X_test, y_test = data.X[5000:], data.y[5000:]
forest = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, random_state=0).fit(X_train, y_train)

section("1. Regression: point in the gap")
reg = make_uncertainty_benchmark(n_samples=4000, random_state=0, task="regression")
reg_forest = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=0).fit(reg.X, reg.y)
expl = explain_uncertainty(reg_forest, reg.queries["gap"], reg.X, imputer="knn", feature_names=names)
print(expl.summary())
plot_uncertainty_explanation(expl).savefig(OUT / "regression_gap_explanation.png", dpi=150)

section("2. MC dropout (sklearn MLP, 20 fixed masks): point in the gap")
mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, random_state=0).fit(X_train, y_train)
dropout = mc_dropout_sklearn(mlp, n_samples=20, p=0.2)
expl = explain_uncertainty(dropout, data.queries["gap"], X_train, imputer="knn", feature_names=names)
print(expl.summary())

section("3. Credal set of the forest: point in the gap")
expl = explain_uncertainty(forest, data.queries["gap"], X_train, measure="credal", imputer="knn", feature_names=names)
print(expl.summary())

section("4. Conformal prediction (alpha = 0.1): what puts both classes into the prediction set?")
conformal = ConformalSetSizeGame(forest, X_test[0], X_train, (X_cal, y_cal), alpha=0.1)
sizes = conformal.set_sizes(X_test)
covered = forest.predict_proba(X_test)[np.arange(len(y_test)), y_test] >= conformal.threshold
two_classes = np.flatnonzero(sizes == 2)
print(f"coverage on test data: {covered.mean():.3f}, mean set size: {sizes.mean():.2f}")
print(f"{len(two_classes)} of {len(X_test)} test points have both classes in their set")
# With two classes the set size at a single point is a coarse quantity (1 or 2),
# so we explain a point that has a two-class set and then average over many of them.
expl = explain_set_size(forest, X_test[two_classes[0]], X_train, (X_cal, y_cal), feature_names=names)
print(expl.summary())
plot_scalar_explanation(expl).savefig(OUT / "conformal_two_class_set.png", dpi=150)
many = [
    explain_set_size(forest, X_test[i], X_train, (X_cal, y_cal), index="SV", n_samples=50).shapley
    for i in two_classes[:40]
]
mean_shapley = np.mean(many, axis=0)
pairs = zip(names, mean_shapley, strict=True)
print("mean Shapley value over 40 such points:", ", ".join(f"{name} {value:+.2f}" for name, value in pairs))

section("5. OOD score: one shifted feature, and the unseen combination")
shifted = np.array([0.3, 0.2, 0.3, 3.0])  # 'dummy' far outside its range of [-1, 1]
expl = explain_ood(shifted, X_train, feature_names=names)
print(expl.summary())
plot_scalar_explanation(expl).savefig(OUT / "ood_shift.png", dpi=150)
expl = explain_ood(data.queries["gap"], X_train, imputer="knn", feature_names=names)
print()
print(expl.summary())
plot_interaction_network(expl).savefig(OUT / "ood_gap_network.png", dpi=150)

section("6. Global view over 150 test points")
# Aggregation B ("pointwise"): the baseline is the average uncertainty on the data, which makes
# the global picture easier to read than the default A (see "Designfragen" in the README).
explainer = UncertaintyExplainer(forest, X_train, index="SV", feature_names=names, aggregation="pointwise")
explanations = explainer.explain_many(X_test[:150])
plot_uncertainty_beeswarm(explanations).savefig(OUT / "synthetic_beeswarm.png", dpi=150)
plot_global_importance(explanations).savefig(OUT / "synthetic_importance.png", dpi=150)

print(f"\nFigures written to {OUT}/")
