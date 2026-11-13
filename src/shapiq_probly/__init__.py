"""shapiq-probly: explain which features make a model uncertain, and why.

Uncertainty measures become value functions of cooperative games over features;
Shapley values and interactions of the total, aleatoric and epistemic games
tell whether a feature contributes noise or missing knowledge.
"""

from .adapters import as_member_predictor, from_members, from_sklearn
from .conformal import ConformalSetSizeGame
from .dropout import mc_dropout_sklearn
from .exact import all_coalitions, compute_index, moebius_transform
from .explainer import UncertaintyExplainer, explain_game, explain_ood, explain_set_size, explain_uncertainty
from .explanation import Attribution, ScalarExplanation, UncertaintyExplanation, global_importance
from .games import ScalarGame, UncertaintyGame
from .imputation import GaussianSampler, KNNSampler, MarginalSampler, Sampler
from .measures import Measure, credal_decomposition, entropy_decomposition, variance_decomposition
from .ood import KNNDistanceScore, OODGame
from .plots import (
    plot_global_importance,
    plot_interaction_network,
    plot_scalar_explanation,
    plot_uncertainty_beeswarm,
    plot_uncertainty_explanation,
    plot_uncertainty_profile,
)
from .synthetic import make_uncertainty_benchmark

__version__ = "0.2.0"

__all__ = [
    "Attribution",
    "ConformalSetSizeGame",
    "GaussianSampler",
    "KNNDistanceScore",
    "KNNSampler",
    "MarginalSampler",
    "Measure",
    "OODGame",
    "Sampler",
    "ScalarExplanation",
    "ScalarGame",
    "UncertaintyExplainer",
    "UncertaintyExplanation",
    "UncertaintyGame",
    "all_coalitions",
    "as_member_predictor",
    "compute_index",
    "credal_decomposition",
    "entropy_decomposition",
    "explain_game",
    "explain_ood",
    "explain_set_size",
    "explain_uncertainty",
    "from_members",
    "from_sklearn",
    "global_importance",
    "make_uncertainty_benchmark",
    "mc_dropout_sklearn",
    "moebius_transform",
    "plot_global_importance",
    "plot_interaction_network",
    "plot_scalar_explanation",
    "plot_uncertainty_beeswarm",
    "plot_uncertainty_explanation",
    "plot_uncertainty_profile",
    "variance_decomposition",
]
