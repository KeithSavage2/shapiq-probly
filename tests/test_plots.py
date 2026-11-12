import matplotlib.pyplot as plt
import pytest

from shapiq_probly import (
    explain_uncertainty,
    plot_interaction_network,
    plot_uncertainty_explanation,
    plot_uncertainty_profile,
)


@pytest.mark.parametrize("mode", ["uncertainty", "information_gain"])
def test_plots_render(toy_model, toy_background, mode, tmp_path):
    expl = explain_uncertainty(toy_model, toy_background[0], toy_background, n_samples=20, mode=mode)
    for i, plot in enumerate((plot_uncertainty_explanation, plot_uncertainty_profile, plot_interaction_network)):
        fig = plot(expl)
        fig.savefig(tmp_path / f"{i}.png")
        assert (tmp_path / f"{i}.png").stat().st_size > 0
        plt.close(fig)


def test_network_needs_pairs(toy_model, toy_background):
    expl = explain_uncertainty(toy_model, toy_background[0], toy_background, n_samples=20, index="SV")
    with pytest.raises(ValueError, match="order-2"):
        plot_interaction_network(expl)
