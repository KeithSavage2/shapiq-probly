import matplotlib.pyplot as plt
import numpy as np
import pytest

from shapiq_probly import (
    UncertaintyExplainer,
    global_importance,
    plot_global_importance,
    plot_uncertainty_beeswarm,
)
from shapiq_probly.plots import _swarm_offsets


@pytest.fixture(scope="module")
def many(benchmark, forest):
    explainer = UncertaintyExplainer(forest, benchmark.X, index="SV", n_samples=30)
    return explainer.explain_many(benchmark.X[:25])


def test_global_importance(many):
    importance = global_importance(many)
    assert set(importance) == {"total", "aleatoric", "epistemic", "prediction"}
    expected = np.mean([np.abs(e.shapley["epistemic"]) for e in many], axis=0)
    np.testing.assert_allclose(importance["epistemic"], expected)
    assert importance["total"].shape == (4,)
    with pytest.raises(ValueError, match="at least one"):
        global_importance([])


def test_swarm_offsets_stay_inside_the_row():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(scale=0.01, size=300), rng.normal(size=20)])
    offsets = _swarm_offsets(values, -3, 3)
    assert np.abs(offsets).max() <= 0.4 + 1e-12
    assert len(np.unique(offsets)) > 10
    assert (_swarm_offsets(np.array([-2.0, 0.0, 2.0]), -3, 3) == 0).all()


def test_global_plots_render(many, reg_benchmark, reg_forest, tmp_path):
    regression = UncertaintyExplainer(reg_forest, reg_benchmark.X, index="SV", n_samples=20).explain_many(
        reg_benchmark.X[:8]
    )
    for explanations in (many, regression):
        for plot in (plot_uncertainty_beeswarm, plot_global_importance):
            fig = plot(explanations)
            fig.savefig(tmp_path / "g.png")
            assert (tmp_path / "g.png").stat().st_size > 0
            plt.close(fig)
    fig = plot_uncertainty_beeswarm(many, notions=("total", "aleatoric", "epistemic"), max_features=2)
    assert len(fig.axes[0].get_yticklabels()) == 2
    plt.close(fig)
