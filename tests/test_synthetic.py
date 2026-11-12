"""Ground-truth checks: does the explanation recover what we built into the data?"""

import numpy as np
import pytest

from shapiq_probly import explain_uncertainty
from shapiq_probly.synthetic import FEATURE_NAMES, in_gap, true_proba

NOISE, PAIR_A, PAIR_B, DUMMY = range(4)


def test_generator(benchmark):
    assert benchmark.X.shape == (3000, 4)
    assert not in_gap(benchmark.X).any()
    assert in_gap(benchmark.queries["gap"]).all()
    assert true_proba(benchmark.queries["noisy"])[0] == pytest.approx(0.5, abs=0.01)
    assert true_proba(benchmark.queries["certain"])[0] > 0.99


@pytest.fixture(scope="module", params=["marginal", "knn"])
def explanations(request, benchmark, forest):
    return {
        name: explain_uncertainty(
            forest, x, benchmark.X, imputer=request.param, n_samples=100, feature_names=list(FEATURE_NAMES)
        )
        for name, x in benchmark.queries.items()
    }


def test_gap_point_is_epistemic_and_noisy_point_is_aleatoric(explanations):
    gap, noisy, certain = (explanations[k].uncertainty for k in ("gap", "noisy", "certain"))
    assert gap["epistemic"] > 3 * certain["epistemic"]
    assert gap["epistemic"] > noisy["epistemic"]
    assert noisy["aleatoric"] > gap["aleatoric"] > certain["aleatoric"]


def test_n1_gap_shows_up_as_the_strongest_epistemic_interaction(explanations):
    expl = explanations["gap"]
    name, value = expl.top_interactions("epistemic", 1)[0]
    assert name == "pair_a x pair_b"
    assert value > 0.1
    assert expl.moebius("epistemic")[(PAIR_A, PAIR_B)] > 0.1


def test_noise_feature_is_the_main_aleatoric_driver_at_the_noisy_point(explanations):
    sv = explanations["noisy"].shapley["aleatoric"]
    assert sv.argmax() == NOISE
    assert sv[NOISE] > 0.05


def test_dummy_feature_is_negligible(explanations):
    for expl in explanations.values():
        for notion in ("total", "aleatoric", "epistemic"):
            sv = np.abs(expl.shapley[notion])
            assert sv[DUMMY] < 0.05
            assert sv[DUMMY] <= sv.max() / 3 or sv.max() < 0.05


def test_additivity_on_real_ensemble(explanations):
    assert all(e.check_additivity() for e in explanations.values())
