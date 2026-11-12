"""Regression case study on the diabetes data set (442 patients, 10 features, bundled with sklearn).

Target: a measure of disease progression one year after baseline. The features
are already standardised by sklearn. A random forest gives a mean and a leaf
variance per tree; the variance decomposition splits the predictive variance
into noise (aleatoric) and disagreement between trees (epistemic).

Run from the repository root:  python examples/regression_diabetes.py
Figures are written to examples/output/.
"""

import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

from shapiq_probly import (
    UncertaintyExplainer,
    global_importance,
    plot_global_importance,
    plot_interaction_network,
    plot_uncertainty_beeswarm,
    plot_uncertainty_explanation,
    plot_uncertainty_profile,
)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

data = load_diabetes()
names = list(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=0)
# min_samples_leaf well above 1: the leaf variance is the tree's estimate of the noise.
forest = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=0).fit(X_train, y_train)
cv = cross_val_score(
    RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=0), data.data, data.target
)
# The data set is small and noisy, so the score depends a lot on the split.
print(f"R^2 on the test split: {forest.score(X_test, y_test):.2f} (5-fold cross-validation: {cv.mean():.2f})")

# ---- local: the test patient with the largest epistemic uncertainty -----------------
explainer = UncertaintyExplainer(forest, X_train, feature_names=names, n_samples=100)
tree_means = np.stack([tree.predict(X_test) for tree in forest.estimators_])
patient = int(tree_means.var(axis=0).argmax())
start = time.time()
expl = explainer.explain(X_test[patient])
print(f"\n=== test patient {patient} (true value {y_test[patient]:.0f}, {time.time() - start:.1f} s) ===")
print(expl.summary())
print("additive:", expl.check_additivity(1e-6))
plot_uncertainty_explanation(expl).savefig(OUT / "diabetes_explanation.png", dpi=150)
plot_uncertainty_profile(expl).savefig(OUT / "diabetes_profile.png", dpi=150)
plot_interaction_network(expl).savefig(OUT / "diabetes_network.png", dpi=150)

# ---- global: which features drive uncertainty across patients? ----------------------
n_global = 60
start = time.time()
global_explainer = UncertaintyExplainer(forest, X_train, feature_names=names, index="SV", n_samples=50)
explanations = global_explainer.explain_many(X_test[:n_global])
print(f"\n=== global view over {n_global} test patients ({time.time() - start:.0f} s) ===")
importance = global_importance(explanations)
for notion in ("aleatoric", "epistemic"):
    ranking = np.argsort(-importance[notion])[:4]
    print(f"{notion:<10}", ", ".join(f"{names[i]} ({importance[notion][i]:.0f})" for i in ranking))
plot_uncertainty_beeswarm(explanations).savefig(OUT / "diabetes_beeswarm.png", dpi=150)
plot_global_importance(explanations).savefig(OUT / "diabetes_importance.png", dpi=150)

print(f"\nFigures written to {OUT}/")
